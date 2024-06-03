import logging
import os
import zipfile
from abc import ABC

import torch
import transformers
from transformers import AutoTokenizer, BloomForCausalLM, BloomTokenizerFast, LlamaForCausalLM
from ts.torch_handler.base_handler import BaseHandler

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.shardformer import ShardConfig
from colossalai.testing import free_port

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)
logger.info("ColossalAI version %s", colossalai.__version__)


class ColossalInferenceHandler(BaseHandler, ABC):
    """
    Transformers handler class for testing
    """

    def __init__(self):
        super(ColossalInferenceHandler, self).__init__()
        self.infer_engine = None
        self.max_batch_size = None
        self.max_input_len = None
        self.max_output_len = None
        self.tokenizer = None
        self.initialized = False

    def initialize(self, ctx):
        """Expected behaviour: the sharded Bloom/Llama model is loaded.

        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        if ctx is not None or not hasattr(ctx, "model_yaml_config"):
            logger.error("Context ctx and model-config are not appropriately passed in.")

        self.manifest = ctx.manifest
        gpu_id = ctx.system_properties.get("gpu_id", -1)
        model_dir = ctx.system_properties.get("model_dir")

        # Inference configs are collected together in model yaml config for handler use
        inference_config = ctx.model_yaml_config["handler"]
        self.inference_config = inference_config
        logger.info(self.inference_config)

        self.tp_size = self.inference_config.get("tp_size", 1)
        self.max_batch_size = self.inference_config.get("max_batch_size", 4)
        self.max_input_len = self.inference_config.get("max_input_len", 1024)
        self.max_output_len = self.inference_config.get("max_output_len", 128)

        self.device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() and gpu_id >= 0 else "cpu")
        logger.info(f"Device set to {self.device}")
        logger.info(f"torch.cuda.device_count() {torch.cuda.device_count()}")

        # Unpacking from model_dir
        model_dir_path = os.path.join(model_dir, "model")
        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir_path)
        logger.info(f"Loading {self.inference_config['model_type']} pretrain model and tokenizer")
        if self.inference_config["model_type"] == "bloom":
            self.model = BloomForCausalLM.from_pretrained(
                model_dir_path,
            )
            self.tokenizer = BloomTokenizerFast.from_pretrained(model_dir_path, return_tensors="pt")
        elif self.inference_config["model_type"] == "llama":
            self.model = LlamaForCausalLM.from_pretrained(
                model_dir_path,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir_path, return_tensors="pt")
        else:
            logger.warning(f"Model type {self.inference_config['model_type']} not supported yet.")

        logger.info("Transformer model from path %s loaded successfully", model_dir)

        # NOTE world_size, rank, host, port here are used to launch colossalai dist environment
        # This world_size is different from the world size of TorchServe
        world_size = int(os.getenv("WORLD_SIZE", self.tp_size))
        assert world_size == 1, "Colossal-Inference with tensor parallel is not supported on TorchServe for now"
        rank = int(os.getenv("RANK", gpu_id))
        local_rank = int(os.getenv("LOCAL_RANK", gpu_id))
        host = os.getenv("MASTER_ADDR", "localhost")
        port = os.getenv("MASTER_PORT", free_port())  # use a random free port

        logger.info(
            f"  world_size {world_size}" f"  local_rank {local_rank}" f"  rank {rank}" f"  host {host}" f"  port {port}"
        )

        torch.cuda.set_device(self.device)
        self.model.half()
        self.model.cuda()
        self.model.eval()

        colossalai.launch(rank=rank, world_size=world_size, host=host, port=port, backend="nccl")
        logger.info("Initializing TPInferEngine ...")
        shard_config = ShardConfig(
            enable_tensor_parallelism=True if self.tp_size > 1 else False, extra_kwargs={"inference_only": True}
        )
        self.infer_engine = TPInferEngine(
            self.model, shard_config, self.max_batch_size, self.max_input_len, self.max_output_len
        )
        logger.info("TPInferEngine initialized successfully")

        self.model = self.infer_engine.model
        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests: The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        logger.info("Pre-processing requests")
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            logger.info("Received text: '%s'", input_text)

            inputs = self.tokenizer.encode_plus(
                input_text,
                max_length=self.max_input_len,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            # making a batch out of the recieved requests
            # attention masks are passed for cases where input tokens are padded.
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat((attention_mask_batch, attention_mask), 0)
        return (input_ids_batch, attention_mask_batch)

    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        input_ids_batch, attention_mask_batch = input_batch
        inferences = []

        do_sample = self.inference_config.get("do_sample", True)
        top_p = self.inference_config.get("top_p", 0.95 if do_sample else 1.0)
        top_k = self.inference_config.get("top_k", 60 if do_sample else 50)
        input_ids_batch = input_ids_batch.to(self.device)
        outputs = self.infer_engine.generate(
            dict(input_ids=input_ids_batch, attention_mask=attention_mask_batch),
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
        )

        for i, _ in enumerate(outputs):
            inferences.append(self.tokenizer.decode(outputs[i], skip_special_tokens=True))

        # For testing only
        logger.info(
            f"Generated text: {inferences}",
        )

        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
