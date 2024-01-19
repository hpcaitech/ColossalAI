import dataclasses
from typing import List, Dict, Any
import json
import os

from transformers import PreTrainedTokenizer
from coati.dataset.utils import (
    find_all_occurrence_subsequence,
    find_first_occurrence_subsequence,
    find_sep_tokens
)
from colossalai.logging import get_dist_logger

logger = get_dist_logger()

DUMMY_SYSTEM_MSG = "Dummy system message"
DUMMY_USER_MSG = "Dummy user message"
DUMMY_ASSISTANT_MSG = "Dummy assistant message"
DUMMY_MSG_WITH_SYSTEM = [
        {
          "role": "system",
          "content": DUMMY_SYSTEM_MSG
        },
        {
          "role": "user",
          "content": DUMMY_USER_MSG
        },
        {
          "role": "assistant",
          "content": DUMMY_ASSISTANT_MSG
        },
        {
          "role": "user",
          "content": DUMMY_USER_MSG
        },
        {
          "role": "assistant",
          "content": DUMMY_ASSISTANT_MSG
        },
        {
          "role": "user",
          "content": DUMMY_USER_MSG
        },
        {
          "role": "assistant",
          "content": DUMMY_ASSISTANT_MSG
        }
      ]
          

@dataclasses.dataclass
class Conversation:
    tokenizer: PreTrainedTokenizer
    system_message: str
    chat_template: str
    human_line_start: List[int] = None # List[int] tokens that indicate the start of a human line
    human_line_end: List[int] = None  # List[int] tokens that indicate the end of a human line
    assistant_line_start: List[int] = None # List[int] tokens that indicate the start of a assistant line
    assistant_line_end: List[int] = None # List[int] tokens that indicate the end of a assistant line
    end_of_system_line_position: int=None # The position of the end of system line in the chat_template

    @classmethod
    def from_config(cls, tokenizer: PreTrainedTokenizer, config: Dict):
        """
        Setup the conversation template from config
        """
        tokenizer.chat_template = config['chat_template']
        conv = cls(tokenizer, config['system_message'], config['chat_template'], config['human_line_start'], config['human_line_end'],
                config['assistant_line_start'], config['assistant_line_end'], config['end_of_system_line_position'])
        conv.clear()
        return conv

    def clear(self):
        self.messages = []

    @classmethod
    def get_conversation_template_keys(cls):
        return ['system_message', 'chat_template', 'human_line_start', 'human_line_end', 'assistant_line_start', 'assistant_line_end', 'end_of_system_line_position']

    def __str__(self):
        return json.dumps({k:self.__dict__[k] for k in self.__dict__ if k not in ['tokenizer', 'messages']}, ensure_ascii=False, indent=4)

    def get_prompt(self, length: int = None, get_seps_info: bool=False, add_generation_prompt=False) -> Any:
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
            messages = [{'role':'system','content':self.system_message}]+self.messages[:length]
        else:
            messages = self.messages[:length]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        if get_seps_info:
            seps_order = []
            for message in self.messages[:length]:
                if message['role'] == 'user':
                    seps_order.append('human_line_start')
                    seps_order.append('human_line_end')
                elif message['role'] == 'assistant':
                    seps_order.append('assistant_line_start')
                    seps_order.append('assistant_line_end')
            return prompt, {'end_of_system_line_position': self.end_of_system_line_position,
                'seps_order': seps_order}
        else:
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
        assert role in ['user', 'assistant']
        self.messages.append({'role': role, 'content': message})

    def copy(self):
        return Conversation(
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            human_line_start=self.human_line_start,
            human_line_end=self.human_line_end,
            assistant_line_start=self.assistant_line_start,
            assistant_line_end=self.assistant_line_end,
        )


def automatically_set_conversation_config(tokenizer: PreTrainedTokenizer, chat_template_config: Dict=None) -> dict:
    """
    Automatically set up the conversation config for the tokenizer with a dummy conversation, if the tokenizer doesn't have a default chat_template,
    raise error to remind the user to set it manually.

    Expect conversation format
    - support chat format only
        [system message]<human_line_start>[human line]<human_line_end><assistant_line_start>[assistant line]<assistant_line_end>...[assistant line]<assistant_line_end>
    check huggingface's doc for more details regarding chat template:
        https://huggingface.co/docs/transformers/main/chat_templating

    Args:
        tokenizer: The tokenizer to use
        chat_template_config: the chat_template_config to use.
    """
    if not isinstance(tokenizer.chat_template, str) or len(tokenizer.chat_template)==0:
        if isinstance(tokenizer.default_chat_template, str) and len(tokenizer.default_chat_template)>0:
            tokenizer.chat_template = tokenizer.default_chat_template
    if 'chat_template' in chat_template_config and chat_template_config['chat_template'] is not None:
        tokenizer.chat_template = chat_template_config['chat_template']
    assert isinstance(tokenizer.chat_template, str) and len(tokenizer.chat_template)>0, \
        "Please set the chat_template of the tokenizer"

    # Generate conversation template config for conversation with Dummy messages
    dummy_chat_messages = DUMMY_MSG_WITH_SYSTEM
    if chat_template_config['system_message'] is not None:
        dummy_chat_messages[0]['content']=chat_template_config['system_message']
    else:
        logger.warning("No system message is provided, if the chat template requires a system message, please provide it.")
        dummy_chat_messages.pop(0)
    prompt = tokenizer.apply_chat_template(dummy_chat_messages, tokenize=False, add_generation_prompt=False)

    # Locate user and assistant line
    occurances_of_user = find_all_occurrence_subsequence(prompt, DUMMY_USER_MSG)
    occurances_of_assistant = find_all_occurrence_subsequence(prompt, DUMMY_ASSISTANT_MSG)
    assert len(occurances_of_user) == len(occurances_of_assistant) == 3
    assert prompt[occurances_of_user[0]+len(DUMMY_USER_MSG):occurances_of_assistant[0]] == \
        prompt[occurances_of_user[1]+len(DUMMY_USER_MSG):occurances_of_assistant[1]] == \
        prompt[occurances_of_user[2]+len(DUMMY_USER_MSG):occurances_of_assistant[2]]
    
    # Calculate the seps with heuristics
    human_line_end_and_assistant_line_start = prompt[occurances_of_user[0]+len(DUMMY_USER_MSG):occurances_of_assistant[0]]
    assert prompt[occurances_of_assistant[0]+len(DUMMY_ASSISTANT_MSG):occurances_of_user[1]] == \
        prompt[occurances_of_assistant[1]+len(DUMMY_ASSISTANT_MSG):occurances_of_user[2]]
    assistant_line_end_and_human_line_start = prompt[occurances_of_assistant[0]+len(DUMMY_ASSISTANT_MSG):occurances_of_user[1]]
    prompt_tail = prompt[occurances_of_assistant[-1]+len(DUMMY_ASSISTANT_MSG):]
    assistant_line_end = ""
    for i in range(len(prompt_tail)):
        if prompt_tail[i]==assistant_line_end_and_human_line_start[i]:
            assistant_line_end = prompt_tail[:i+1]
    human_line_start = assistant_line_end_and_human_line_start[len(assistant_line_end):].strip()
    assistant_line_end = assistant_line_end.strip()
    human_line_end = human_line_end_and_assistant_line_start.strip()
    assistant_line_start = "" # Note that usually assistant line start doesn't matter if human_line_end already include it
    end_of_system_line_position = len(tokenizer([prompt[:occurances_of_user[0]]], add_special_tokens=False)["input_ids"][0])-len(human_line_start)
    conversation_template_config = {
        "chat_template": tokenizer.chat_template,
        "system_message": chat_template_config['system_message'],
        "human_line_start": [],
        "human_line_end": [],
        "assistant_line_start": [],
        "assistant_line_end": [],
        "end_of_system_line_position": end_of_system_line_position
    }

    # Find the seps tokens
    conversation_template_config['human_line_start'] = find_sep_tokens(prompt, tokenizer, "human_line_start", 
                                                                human_line_start, conversation_template_config)
    conversation_template_config['human_line_end'] = find_sep_tokens(prompt, tokenizer, "human_line_end", 
                                                                human_line_end, conversation_template_config)
    conversation_template_config['assistant_line_start'] = find_sep_tokens(prompt, tokenizer, "assistant_line_start", 
                                                                assistant_line_start, conversation_template_config)
    conversation_template_config['assistant_line_end'] = find_sep_tokens(prompt, tokenizer, "assistant_line_end", 
                                                                assistant_line_end, conversation_template_config)
    return conversation_template_config


def setup_conversation_template(tokenizer: PreTrainedTokenizer, chat_template_config: Dict=None, save_path: str=None) -> Conversation:
    """
    Setup the conversation template, if chat_template is given, will replace the default chat_template of the tokenizer
    with it. Otherwise, the default chat_template will be used. If the tokenizer doesn't have a default chat_template,
    raise error to remind the user to set it manually.

    Args:
        tokenizer: The tokenizer to use
        chat_template_config: 
            {
                "system_message": str The system message to use
                "chat_template": str The chat_template to use, if None, will use the default chat_template of the tokenizer
                                if you want to use custom seps, please set the chat_template and the seps argument
                "human_line_start": List[int] tokens that indicate the start of a human line,
                "human_line_end": List[int] tokens that indicate the end of a human line,
                "assistant_line_start": List[int] tokens that indicate the start of a assistant line,
                "assistant_line_end": List[int]  tokens that indicate the end of a assistant line
                "end_of_system_line_position": int For some prompt sequence control tokens may appear in system message,
                                This field defines the index of the last token in the system message
            }
    """
    if any([s not in chat_template_config.keys() for s in Conversation.get_conversation_template_keys()]):
        # Try to automatically set up conversation template, if fail, it throws an error that you need to do it manually
        assert "system_message" in chat_template_config, "Please provide system message."
        logger.info("No conversation template config is provided or incomplete, will try generating the conversation tempalte config automatically.")
        conversation_template_config = automatically_set_conversation_config(tokenizer, chat_template_config)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf8') as f:
            logger.info(f"Successfully generated a conversation tempalte config, save to {save_path}.")
            json.dump(conversation_template_config, f, indent=4, ensure_ascii=False)
        return Conversation.from_config(tokenizer, conversation_template_config)
    else:
        # Setup conversation manually
        return Conversation.from_config(tokenizer, chat_template_config)
