from abc import abstractclassmethod
from typing import Dict, List

from colossal_eval.utils import Conversation, prompt_templates

from colossalai.logging import DistributedLogger


class BaseModel:
    """
    Base class for model wrapper.

    Args:
        path: The path to the model.
        model_max_length: The maximum sequence length of the model.
        prompt_template: The model's prompt template.
        batch_size: Batch size for inference.
        logger: Logger for the model.
    """

    def __init__(
        self,
        path: str,
        model_max_length: int = 2048,
        prompt_template: Conversation = None,
        batch_size: int = 1,
        logger: DistributedLogger = None,
    ):
        self.path = path
        self.model_max_length = model_max_length

        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = prompt_templates["plain"]

        self.batch_size = batch_size
        self.logger = logger

    @abstractclassmethod
    def inference(self, data: List[Dict]) -> None:
        """
        Infer the given data.
        This function will call self.generate() to get model outputs and also self.model(input) to get logits.

        Args:
            data: The data for inference.
        """

    @abstractclassmethod
    def generate(self, inputs: List[str], max_new_tokens: int) -> List[str]:
        """
        Generate results given a list of inputs.

        Args:
            inputs: A list of strings.
            max_new_tokens: The maximum length of the output.

        Returns:
            A list of generated strings.
        """

    @abstractclassmethod
    def get_loss(self, batch: List[str], batch_target: List[str]) -> List[float]:
        """
        Get loss given batch and batch with target.
        Use their length difference after tokenization to mask the loss and only compute loss at target tokens.

        Args:
            batch: batch prompt without target answer.
            batch_target: batch prompt with target answer.

        Returns:
            A list of loss.
        """

    def to(self, device):
        self.model.to(device)
