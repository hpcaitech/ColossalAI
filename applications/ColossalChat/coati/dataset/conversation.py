import dataclasses
import json
import os
from typing import Any, Dict, List

import torch.distributed as dist
from transformers import AutoTokenizer, PreTrainedTokenizer

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


@dataclasses.dataclass
class Conversation:
    tokenizer: PreTrainedTokenizer
    system_message: str
    chat_template: str
    stop_ids: List[int]
    end_of_assistant: str
    roles = ["user", "assistant"]

    @classmethod
    def from_config(cls, tokenizer: PreTrainedTokenizer, config: Dict):
        """
        Setup the conversation template from config
        """
        tokenizer.chat_template = config["chat_template"]
        conv = cls(
            tokenizer, config["system_message"], config["chat_template"], config["stop_ids"], config["end_of_assistant"]
        )
        conv.clear()
        return conv

    def clear(self):
        self.messages = []

    @classmethod
    def get_conversation_template_keys(cls):
        return ["system_message", "chat_template"]

    def __str__(self):
        return json.dumps(
            {k: self.__dict__[k] for k in self.__dict__ if k not in ["tokenizer", "messages"]},
            ensure_ascii=False,
            indent=4,
        )

    def get_prompt(self, length: int = None, add_generation_prompt=False) -> Any:
        """
        Retrieves the prompt for the conversation.

        Args:
            length (int, optional): The number of messages to include in the prompt. Defaults to None.
            get_seps_info (bool, optional): Whether to include separator information in the output. Defaults to False.
            add_generation_prompt (bool, optional): Whether to add the assistant line start token in generation (for generation only). Defaults to False.

        Returns:
            str or tuple: The prompt string if get_seps_info is False, otherwise a tuple containing the prompt string and separator information.
        """

        if length is None:
            length = len(self.messages)

        assert length <= len(self.messages)
        if self.system_message is not None:
            messages = [{"role": "system", "content": self.system_message}] + self.messages[:length]
        else:
            messages = self.messages[:length]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return prompt

    def save_prompt(self):
        return self.get_prompt()

    def append_message(self, role: str, message: str):
        """
        Append a message to the conversation.

        Args:
            role (str): The role of the message sender. Must be either 'user' or 'assistant'.
            message (str): The content of the message.

        Raises:
            AssertionError: If the role is not 'user' or 'assistant'.
        """
        assert role in self.roles
        self.messages.append({"role": role, "content": message})

    def copy(self):
        return Conversation(tokenizer=self.tokenizer, chat_template=self.chat_template)


def setup_conversation_template(
    tokenizer: PreTrainedTokenizer, chat_template_config: Dict = None, save_path: str = None
) -> Conversation:
    """
    Setup the conversation template, if chat_template is given, will replace the default chat_template of the tokenizer
    with it. Otherwise, the default chat_template will be used. If the tokenizer doesn't have a default chat_template,
    raise error to remind the user to set it manually.

    Args:
        tokenizer: The tokenizer to use
        chat_template_config:
            {
                "system_message": str The system message to use
                "chat_template": str The chat_template to use, if can be a chat_template, a huggingface model path or a local model.
                    if a huggeface model path or a local model, the chat_template will be loaded from the model's tokenizer's default chat template.
                "stop_ids": List[int], the token ids used to terminate generation. You need to provide this for ppo training and generation.
            }
    """
    if any([s not in chat_template_config.keys() for s in Conversation.get_conversation_template_keys()]):
        # Try to automatically set up conversation template, if fail, it throws an error that you need to do it manually
        if "end_of_assistant" not in chat_template_config:
            raise ValueError("Please set the end of assistant token.")
        if "system_message" not in chat_template_config:
            logger.warning("No system message is provided, will not use system message.")
        if "chat_template" not in chat_template_config:
            logger.warning("No chat_template is provided, will try to load it from the tokenizer.")
            if tokenizer.chat_template != None:
                chat_template_config["chat_template"] = tokenizer.chat_template
            else:
                raise ValueError(
                    f"Load a tokenizer from {chat_template_config['chat_template']}, which doesn't have a default chat template, please set it manually."
                )
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(chat_template_config["chat_template"])
                if tokenizer.chat_template != None:
                    chat_template_config["chat_template"] = tokenizer.chat_template
                else:
                    raise ValueError(
                        f"Load a tokenizer from {chat_template_config['chat_template']}, which doesn't have a default chat template, please set it manually."
                    )
                logger.warning(
                    f"chat_template is provided as a local model path or huggingface model path, loaded chat_template from \"{chat_template_config['chat_template']}\"."
                )
            except OSError:
                pass
            except ValueError as e:
                raise ValueError(e)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf8") as f:
            logger.info(f"Successfully generated a conversation tempalte config, save to {save_path}.")
            json.dump(chat_template_config, f, indent=4, ensure_ascii=False)
    return Conversation.from_config(tokenizer, chat_template_config)
