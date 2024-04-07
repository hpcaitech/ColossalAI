from typing import Any, Optional

from pydantic import BaseModel


# make it singleton
class NumericIDGenerator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NumericIDGenerator, cls).__new__(cls)
            cls._instance.current_id = 0
        return cls._instance

    def __call__(self):
        self.current_id += 1
        return self.current_id


id_generator = NumericIDGenerator()


class ChatMessage(BaseModel):
    role: str
    content: Any


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[Any] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    message: DeltaMessage
