from itertools import count
from typing import List, Tuple, Type, Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torch import distributed as dist

from colossalai.accelerator import get_accelerator
from colossalai.cluster import ProcessGroupMesh
from colossalai.inference.config import DiffusionGenerationConfig, InferenceConfig, ModelShardInferenceConfig
from colossalai.inference.modeling.layers.diffusion import DiffusionPipe
from colossalai.inference.modeling.policy import model_policy_map
from colossalai.inference.struct import DiffusionSequence
from colossalai.inference.utils import get_model_size, get_model_type
from colossalai.logging import get_dist_logger
from colossalai.shardformer.policies.base_policy import Policy

from .base_engine import BaseEngine
from .request_handler import NaiveRequestHandler

PP_AXIS, TP_AXIS = 0, 1


class DiffusionEngine(BaseEngine):
    def __init__(
        self,
        model_or_path: DiffusionPipeline | str,
        inference_config: InferenceConfig = None,
        verbose: bool = False,
        model_policy: Policy | type[Policy] = None,
    ) -> None:
        self.inference_config = inference_config
        self.dtype = inference_config.dtype
        self.high_precision = inference_config.high_precision

        self.verbose = verbose
        self.logger = get_dist_logger(__name__)
        self.model_shard_infer_config = inference_config.to_model_shard_inference_config()

        self.model_type = get_model_type(model_or_path=model_or_path)

        self.init_model(model_or_path, model_policy, self.model_shard_infer_config)

        self.request_handler = NaiveRequestHandler()

        self.counter = count()

        self._verify_args()

    def _verify_args(self) -> None:
        assert isinstance(self.model, DiffusionPipe), "model must be DiffusionPipe"

    def init_model(
        self,
        model_or_path: Union[str, nn.Module, DiffusionPipeline],
        model_policy: Union[Policy, Type[Policy]] = None,
        model_shard_infer_config: ModelShardInferenceConfig = None,
    ):
        """
        Shard model or/and Load weight

        Args:
            model_or_path Union[nn.Module, str]: path to the checkpoint or model of transformer format.
            model_policy (Policy): the policy to replace the model.
            model_inference_config: the configuration for modeling initialization when inference.
            model_shard_infer_config (ModelShardInferenceConfig): the configuration for init of module when inference.
        """
        if isinstance(model_or_path, str):
            model = DiffusionPipeline.from_pretrained(model_or_path, torch_dtype=self.dtype)
            policy_map_key = model.__class__.__name__
            model = DiffusionPipe(model)
        elif isinstance(model_or_path, DiffusionPipeline):
            policy_map_key = model_or_path.__class__.__name__
            model = DiffusionPipe(model_or_path)
        else:
            self.logger.error(f"model_or_path support only str or DiffusionPipeline currently!")

        torch.cuda.empty_cache()
        init_gpu_memory = torch.cuda.mem_get_info()[0]

        self.device = get_accelerator().get_current_device()
        if self.verbose:
            self.logger.info(f"the device is {self.device}")

        if self.verbose:
            self.logger.info(
                f"Before the shard, Rank: [{dist.get_rank()}], model size: {get_model_size(model)} GB, model's device is: {model.device}"
            )

        if model_policy is None:
            model_policy = model_policy_map.get(policy_map_key)

        if not isinstance(model_policy, Policy):
            try:
                model_policy = model_policy()
            except Exception as e:
                raise ValueError(f"Unable to instantiate model policy: {e}")

        assert isinstance(model_policy, Policy), f"Invalid type of model policy: {type(model_policy)}"
        pg_mesh = ProcessGroupMesh(self.inference_config.pp_size, self.inference_config.tp_size)
        tp_group = pg_mesh.get_group_along_axis(TP_AXIS)

        self.model = self._shardformer(
            model,
            model_policy,
            model_shard_infer_config,
            None,
            tp_group=tp_group,
        )

        self.model = model.to(self.device)

        if self.verbose:
            self.logger.info(
                f"After the shard, Rank: [{dist.get_rank()}], model size: {get_model_size(self.model)} GB, model's device is: {model.device}"
            )

        free_gpu_memory, _ = torch.cuda.mem_get_info()
        peak_memory = init_gpu_memory - free_gpu_memory
        if self.verbose:
            self.logger.info(
                f"Rank [{dist.get_rank()}], Model Weight Max Occupy {peak_memory / (1024 ** 3)} GB, Model size: {get_model_size(self.model)} GB"
            )

    def generate(
        self,
        request_ids: Union[List[int], int] = None,
        prompts: Union[List[str], str] = None,
        generation_config: DiffusionGenerationConfig = None,
        **kwargs,
    ) -> Union[List[Union[str, List[PIL.Image.Image], np.ndarray]], Tuple[List[str], List[List[int]]]]:
        """ """
        gen_config_dict = generation_config.to_dict() if generation_config is not None else {}
        prompts = [prompts] if isinstance(prompts, str) else prompts
        request_ids = [request_ids] if isinstance(request_ids, int) else request_ids

        with torch.inference_mode():
            if prompts is not None:
                self.add_request(
                    request_ids=request_ids,
                    prompts=prompts,
                    **gen_config_dict,
                    **kwargs,
                )

            output_reqs_list = []

            # intuition: If user provide a generation config, we should replace the existing one.
            if generation_config is not None:
                self.generation_config = generation_config
                self.generation_config_dict = gen_config_dict

            while self.request_handler.check_unfinished_reqs():
                output_reqs_list += self.step()

            return output_reqs_list

    def add_request(
        self,
        prompts: Union[List[str], str],
        request_ids: Union[List[int], int] = None,
        **kwargs,
    ):
        if request_ids is not None and not isinstance(request_ids, list):
            request_ids = [request_ids]

        if not isinstance(prompts, list):
            prompts = [prompts]

        generation_config = DiffusionGenerationConfig.from_kwargs(**kwargs)
        prompts_num = len(prompts)
        for i in range(prompts_num):
            if request_ids:
                assert isinstance(
                    request_ids[0], int
                ), f"The request_id type must be int, but got {type(request_ids[0])}"
                assert len(request_ids) == prompts_num
                request_id = request_ids[i]
            else:
                request_id = next(self.counter)

            seq = DiffusionSequence(request_id=request_id, prompt=prompts[i], generation_config=generation_config)

            self.request_handler.add_sequence(seq)

    def step(self) -> List[PIL.Image.Image]:
        """
        In each step, do the follows:
            1. Run RequestHandler.schedule() and get the batch used for inference.
            2. run forward to get List[Image]
        Returns:
            List[PIL.Image.Image]: Image Generated by one step.
        """

        input = self.request_handler.schedule()
        ret = self.model(prompt=input.prompt, **input.generation_config.to_dict())
        return ret
