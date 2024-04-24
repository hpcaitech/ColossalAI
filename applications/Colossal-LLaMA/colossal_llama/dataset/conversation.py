#    Copyright 2023 lm-sys@FastChat
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import dataclasses
from enum import Enum, auto
from typing import List


class SeparatorStyle(Enum):
    ADD_BOS_EOS_TOKEN = auto()


@dataclasses.dataclass
class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle
    seps: List[str]

    def clear(self):
        self.messages = []

    def get_prompt(self, length: int = None):
        if length is None:
            length = len(self.messages)

        if self.sep_style == SeparatorStyle.ADD_BOS_EOS_TOKEN:
            ret = self.system
            for role, message in self.messages[0:length]:
                if message:
                    ret += role + ": " + self.seps[0] + message + self.seps[1]
                else:
                    ret += role + ": " + self.seps[0]
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def save_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_BOS_EOS_TOKEN:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + self.seps[0] + message + self.seps[1] + "\n"
                else:
                    ret += role + ": " + self.seps[0]
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
            seps=self.seps,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "seps": self.seps,
        }


LLaMA2_Conv = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.ADD_BOS_EOS_TOKEN,
    seps=["<s>", "</s>"],
)

LLaMA3_Conv = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.ADD_BOS_EOS_TOKEN,
    seps=["<|begin_of_text|>", "<|end_of_text|>"],
)

default_conversation = LLaMA3_Conv
