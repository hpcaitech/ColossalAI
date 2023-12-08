from itertools import count
from logging import Logger
from typing import List, Optional, Union

from transformers import AutoConfig, GenerationConfig

from .config import InferenceConfig
from .get_tokenizer import get_tokenizer
from .inference_struct import BatchHandler, Sequence
from .init_model import init_model


class InferenceEngine:

    """
        InferenceEngine which manages the inference process.
    .

        Args:
            inference_config (Optional[InferenceConfig], optional): Store the configuration information related to inference.
            verbose (bool): Determine whether or not to log the generation process.
    """

    def __init__(
        self,
        inference_config: Optional[InferenceConfig] = None,
        verbose: bool = False,
    ) -> None:
        assert inference_config, "Please provide inference_config."

        # TODO cache_config may need to be modified later.
        # self.request_handler = RequestHandler(cache_config)
        self.tokenizer = get_tokenizer(
            inference_config.tokenizer,
            use_fast_tokenizer=inference_config.use_fast_tokenizer,
            trust_remote_code=inference_config.trust_remote_code,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inference_config = inference_config
        self.hf_model_config = AutoConfig.from_pretrained(
            inference_config.model,
            trust_remote_code=inference_config.trust_remote_code,
            revision=inference_config.revision,
        )
        if verbose:
            self.logger = Logger()

        self._init_model()

        # Will be deleted later.
        self.batch = BatchHandler.init_batch([])

        self.counter = count()
        self._verify_config()

    def _init_model(self):
        """
        Initialize model and distributed training environment(if needed).
        """
        self.model = init_model(self.inference_config, self.hf_model_config)

    def _verify_config(self):
        """
        Verify the configuration to avoid potential bugs.
        """

    def generate(
        self,
        prompts: Union[str, List[str]] = None,
        prompts_token_ids: List[List[int]] = None,
        generation_config: GenerationConfig = None,
    ) -> List[str]:
        """Handling input prompts and executing the inference step.

        This function will handle input prompts, add them to the batch,
        and then proceed with the inference steps until the batch is empty.

        Args:
            prompts (Union[str, List[str]], optional): Input prompts. Defaults to None.
            prompts_token_ids (List[List[int]], optional): token ids of input prompts. Defaults to None.
            generation_config (GenerationConfig, optional): Huggingface GenerationConfig used for inference. Defaults to None.

        Returns:
            List[str]: Inference result returned by one generation.
        """

        prompts_num = None
        if prompts == None:
            assert prompts_token_ids is not None, "When prompts is None, prompts_token_ids must be set."
            prompts_num = len(prompts_token_ids)
        else:
            prompts_num = len(prompts)

        for i in range(prompts_num):
            if prompts == None:
                self.add_request(next(self.counter), None, prompts_token_ids[i])
            else:
                if prompts_token_ids == None:
                    self.add_request(next(self.counter), prompts[i], None)
                else:
                    self.add_request(next(self.counter), prompts[i], prompts_token_ids[i])

        output_list = []

        while not self.batch.is_empty():
            output_list += self.step()

        return output_list

    def add_request(
        self,
        request_id: int = None,
        prompt: str = None,
        input_token_ids: List[int] = None,
    ):
        """Add a single request.

        Args:
            request_id (int, optional): The request ID. Defaults to None.
            prompt (str, optional): The input prompt. Defaults to None.
            input_token_ids (List[int], optional): Token IDs of input prompt. Defaults to None.
        """

        if input_token_ids == None:
            assert prompt, "When the input_token_ids is none, the prompt must be provided."
            input_token_ids = self.tokenizer.encode(prompt)

        block_size = self.inference_config.block_size
        sample_params = None
        block_table_index = 0

        sequence = Sequence(request_id, prompt, input_token_ids, block_size, sample_params, block_table_index)

        # self.batch = self.scheduler.add_seq_group(sequence)
        # Batch here will be returned by the scheduler later.

        self.batch.add_seqs([sequence])

    def step(self) -> List[str]:
        """
        In each step, do the follows:
            1. check whether there are any unfinished reasoning tasks.
            2. Run model to generate the next token

        Returns:
            List[str]: Inference result returned by one step.
        """

        output_list = []
        # Scheduler will return batch and finished sequences later.
        # self.batch, finished_sequences = self.scheduler()

        # The code below is only used for test and will be deleted if scheduler function code is updated.
        finished_sequences = []
        for seq in self.batch.sequences_set:
            finished_sequences.append(seq)

        self.batch.clear_batch()

        # Process the output of completed sentences.
        for seq in finished_sequences:
            if seq.prompt:
                output_str = self.tokenizer.decode(seq.output_token_id, skip_special_tokens=True)
                output_list.append(seq.prompt + output_str)
            else:
                output_str = self.tokenizer.decode(seq.input_token_id + seq.output_token_id, skip_special_tokens=True)
                output_list.append(output_str)

        return output_list
