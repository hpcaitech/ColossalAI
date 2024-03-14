import asyncio
from typing import AsyncIterator

from colossalai.inference.core.async_engine import AsyncInferenceEngine

from .utils import id_generator


async def completion_stream_generator(
    result_generator: AsyncIterator,
):
    return await result_generator


class CompletionServing:
    def __init__(self, engine: AsyncInferenceEngine, served_model: str):
        self.engine = engine
        self.served_model = served_model

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass

    async def create_completion(self, request, generation_config):
        request_dict = await request.json()
        request_id = id_generator()

        prompt = request_dict.pop("prompt")
        stream = request_dict.pop("stream", False)

        # it is not a intuitive way
        self.engine.engine.generation_config = generation_config
        result_generator = self.engine.generate(request_id, prompt=prompt)
        if stream == True:
            return completion_stream_generator(result_generator=result_generator)

        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await self.engine.abort(request_id)
            raise RuntimeError("Client disconnected")

        final_res = await result_generator
        return final_res
