from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from colossalai.cluster import ProcessGroupMesh
from colossalai.inference.config import ModelShardInferenceConfig
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.base_policy import Policy


class BaseEngine(ABC):
    @abstractmethod
    def __init__(self, model_or_path, inference_config=None, verbose=False, model_policy=None):
        pass

    @abstractmethod
    def init_model(self, model_or_path, model_policy=None, model_shard_infer_config=None):
        """
        Init Model for Engine
        """

    @abstractmethod
    def generate(self, request_ids=None, prompts=None, generation_config=None, **kwargs):
        """
        Generate ouptput for coming requests
        """

    @abstractmethod
    def add_request(self, prompts, request_ids=None, **kwargs):
        """
        Add new request to Engine
        """

    @abstractmethod
    def step(self):
        """
        Perform one new step forward
        """

    @abstractmethod
    def _verify_args(self):
        """
        Verify the parameters and members of class
        """

    @torch.inference_mode()
    def capture_model(self):
        """
        Use cuda graph to capture model
        """
        return NotImplementedError("This method should be implemented by subclasses")

    def _shardformer(
        self,
        model: nn.Module,
        model_policy: Policy,
        model_shard_infer_config: ModelShardInferenceConfig = None,
        stage_manager: PipelineStageManager = None,
        tp_group: ProcessGroupMesh = None,
        **kwargs,
    ) -> nn.Module:
        """
        Initialize ShardConfig and replace the model with shardformer.

        Args:
            model (nn.Module): Path or nn.Module of this model.
            model_policy (Policy): The policy to shardformer model which is determined by the model type.
            stage_manager (PipelineStageManager, optional): Used to manage pipeline stages. Defaults to None.
            tp_group (ProcessGroupMesh, optional): Used to manage the process TP group mesh. Defaults to None.

        Returns:
            nn.Module: The model optimized by Shardformer.
        """

        shardconfig = ShardConfig(
            tensor_parallel_process_group=tp_group,
            pipeline_stage_manager=stage_manager,
            enable_tensor_parallelism=(self.inference_config.tp_size > 1),
            enable_fused_normalization=False,
            enable_all_optimization=False,
            enable_flash_attention=False,
            enable_jit_fused=False,
            enable_sequence_parallelism=False,
            extra_kwargs={"model_shard_infer_config": model_shard_infer_config, **kwargs},
        )
        shardformer = ShardFormer(shard_config=shardconfig)
        shard_model, _ = shardformer.optimize(model, model_policy)
        return shard_model
