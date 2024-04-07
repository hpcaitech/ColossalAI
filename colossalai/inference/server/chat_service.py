import asyncio
import codecs
import logging

from fastapi import Request

from colossalai.inference.core.async_engine import AsyncInferenceEngine

from .utils import ChatCompletionResponseStreamChoice, ChatMessage, DeltaMessage, id_generator

logger = logging.getLogger("colossalai-inference")


class ChatServing:
    def __init__(
        self, engine: AsyncInferenceEngine, served_model: str, tokenizer, response_role: str, chat_template=None
    ):
        self.engine = engine
        self.served_model = served_model
        self.tokenizer = tokenizer
        self.response_role = response_role
        self._load_chat_template(chat_template)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass

    async def create_chat(self, request: Request, generation_config):
        request_dict = await request.json()
        messages = request_dict["messages"]
        stream = request_dict.pop("stream", "false").lower()
        add_generation_prompt = request_dict.pop("add_generation_prompt", False)
        request_id = id_generator()
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception as e:
            raise RuntimeError(f"Error in applying chat template from request: {str(e)}")

        # it is not a intuitive way
        self.engine.engine.generation_config = generation_config
        result_generator = self.engine.generate(request_id, prompt=prompt)

        if stream == "true":
            return self.chat_completion_stream_generator(request, request_dict, result_generator, request_id)
        else:
            return await self.chat_completion_full_generator(request, request_dict, result_generator, request_id)

    async def chat_completion_stream_generator(self, request, request_dict, result_generator, request_id: int):
        # Send first response for each request.n (index) with the role
        role = self.get_chat_request_role(request, request_dict)
        n = request_dict.get("n", 1)
        echo = request_dict.get("echo", "false").lower()
        for i in range(n):
            choice_data = ChatCompletionResponseStreamChoice(index=i, message=DeltaMessage(role=role))
            data = choice_data.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"

        # Send response to echo the input portion of the last message
        if echo == "true":
            last_msg_content = ""
            if (
                request_dict["messages"]
                and isinstance(request_dict["messages"], list)
                and request_dict["messages"][-1].get("content")
                and request_dict["messages"][-1].get("role") == role
            ):
                last_msg_content = request_dict["messages"][-1]["content"]
            if last_msg_content:
                for i in range(n):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i, message=DeltaMessage(content=last_msg_content)
                    )
                    data = choice_data.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

        result = await result_generator
        choice_data = DeltaMessage(content=result.output)
        data = choice_data.model_dump_json(exclude_unset=True, exclude_none=True)
        yield f"data: {data}\n\n"

        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: Request,
        request_dict: dict,
        result_generator,
        request_id,
    ):
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await self.engine.abort(request_id)
            return {"error_msg": "Client disconnected"}

        result = await result_generator
        assert result is not None
        role = self.get_chat_request_role(request, request_dict)
        choice_data = ChatMessage(role=role, content=result.output)
        echo = request_dict.get("echo", "false").lower()

        if echo == "true":
            last_msg_content = ""
            if (
                request.messages
                and isinstance(request.messages, list)
                and request.messages[-1].get("content")
                and request.messages[-1].get("role") == role
            ):
                last_msg_content = request.messages[-1]["content"]

            full_message = last_msg_content + choice_data.content
            choice_data.content = full_message

        return choice_data

    def get_chat_request_role(self, request: Request, request_dict: dict) -> str:
        add_generation_prompt = request_dict.get("add_generation_prompt", False)
        if add_generation_prompt:
            return self.response_role
        else:
            return request_dict["messages"][-1]["role"]

    def _load_chat_template(self, chat_template):
        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    self.tokenizer.chat_template = f.read()
            except OSError:
                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                self.tokenizer.chat_template = codecs.decode(chat_template, "unicode_escape")

            logger.info(f"Using supplied chat template:\n{self.tokenizer.chat_template}")
        elif self.tokenizer.chat_template is not None:
            logger.info(f"Using default chat template:\n{self.tokenizer.chat_template}")
        else:
            logger.warning("No chat template provided. Chat API will not work.")
