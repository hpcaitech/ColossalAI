import asyncio

from colossalai.inference.core.async_engine import AsyncInferenceEngine

from .utils import id_generator


class ColossalAICompletionServing:
    def __init__(self, engine: AsyncInferenceEngine, served_model: str):
        self.engine = engine
        self.served_model = served_model

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass

    async def create_completion(self, request):
        request_dict = await request.json()
        request_id = id_generator()
        prompt = request_dict.pop("prompt")

        result_generator = self.engine.generate(request_id, prompt=prompt)

        final_res = None
        async for res in result_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res

        return final_res
