import dataclasses
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer


class SeparatorStyle(Enum):
    ADD_BOS_EOS_TOKEN = auto()
    ALPACA = auto()
    PLAIN = auto()


@dataclasses.dataclass
class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.ADD_BOS_EOS_TOKEN
    sep: str = "</s>"

    def clear(self):
        self.messages = []

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_BOS_EOS_TOKEN:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + self.sep
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ALPACA:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.PLAIN:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += message
                else:
                    ret += ""
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def get_prompt_with_target(self, target):
        prompt = self.get_prompt()
        prompt_with_target = []

        # Some dataset provides multiple target answers.
        # This will make it difficult when we calculate loss.
        # We convert target into list[str] first if the question only has one target answer.
        target_answers = []
        if isinstance(target, str):
            target_answers = [target]
        else:
            target_answers = target

        for target_answer in target_answers:
            if self.sep_style == SeparatorStyle.ADD_BOS_EOS_TOKEN:
                prompt_with_target.append(prompt + target_answer)
            elif self.sep_style == SeparatorStyle.ALPACA:
                prompt_with_target.append(prompt + target_answer)
            elif self.sep_style == SeparatorStyle.PLAIN:
                prompt_with_target.append(prompt + target_answer)
            else:
                raise ValueError(f"Invalid style: {self.sep_style}")

        return prompt_with_target

    def save_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_BOS_EOS_TOKEN:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>\n"
                else:
                    ret += role + ": " + "<s>"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep_style": self.sep_style,
            "sep": self.sep,
        }


def get_few_shot_prefix(
    conv: Conversation, few_shot_data: List[str], tokenizer: Optional[AutoTokenizer], language: str, max_tokens: int
) -> str:
    """
    Get few shot prefix.

    Args:
        conv: Conversation template.
        few_shot_examples: Few shot examples to generate few shot prompt prefix.

    Returns:
        Few shot prompt prefix.
    """

    if language == "English":
        few_shot_prefix = f"The following are answers for questions in an exam.\n\n"
    elif language == "Chinese":
        few_shot_prefix = f"以下是考试中各个问题的答案。\n\n"

    output = None
    for i in range(len(few_shot_data)):
        few_shot_prefix = few_shot_prefix + few_shot_data[i] + "\n\n"

        if len(tokenizer([few_shot_prefix]).input_ids[0]) <= max_tokens:
            output = few_shot_prefix
        else:
            break

    return output if output is not None else few_shot_prefix


def get_batch_prompt(
    conv: Conversation,
    batch: List[Dict],
    few_shot_data: List[str],
    tokenizer: Optional[AutoTokenizer],
    language: Optional[str],
    model_max_length: Optional[int],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Get batch prompt and target.

    Args:
        conv: Conversation template.
        batch: Batch data to generate prompt from.
        few_shot_data: Few shot data to generate few shot prompt prefix.

    Returns:
        Tuple containg batch prompt and target.

    """

    batch_prompt = []
    batch_target = []

    if isinstance(batch[0], dict):
        for b in batch:
            few_shot_prefix = ""
            if few_shot_data is not None:
                # For few-shot, only need input. Otherwise use instruction (in AGIEval).
                query_text = b["input"] if b.get("input", "") != "" else b["instruction"]

                if isinstance(b["target"], str):
                    zero_shot_prompt = query_text + b["target"]
                    max_tokens = model_max_length - len(tokenizer([zero_shot_prompt]).input_ids[0])
                else:
                    raise Exception("When using few-shot, target answer should be a string.")

                few_shot_prefix = get_few_shot_prefix(conv, few_shot_data, tokenizer, language, max_tokens)
            else:
                query_text = b["instruction"] + "\n\n" + b["input"] if b.get("input", "") != "" else b["instruction"]

            conv.append_message(conv.roles[0], few_shot_prefix + query_text)
            conv.append_message(conv.roles[1], None)

            batch_prompt.append(conv.get_prompt())

            target = b["target"]
            if isinstance(b["target"], str):
                target = [target]

            batch_target.append(target)

            conv.clear()

    return batch_prompt, batch_target


conv_coati = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.ADD_BOS_EOS_TOKEN,
    sep="</s>",
)

conv_alpaca = Conversation(
    system="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    roles=("### Instruction", "### Response"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.ALPACA,
    sep="\n\n",
)

conv_plain = Conversation(
    system="",
    roles=("", ""),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="",
)

prompt_templates = {"coati": conv_coati, "alpaca": conv_alpaca, "plain": conv_plain}
