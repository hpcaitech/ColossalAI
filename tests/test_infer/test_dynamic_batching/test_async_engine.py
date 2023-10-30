import asyncio
import os
import uuid

import pytest

import colossalai
from colossalai.inference.async_engine import Async_Engine
from colossalai.inference.dynamic_batching.ray_init_config import RayInitConfig
from colossalai.inference.dynamic_batching.sampling_params import SamplingParams
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

PATH = "config.yaml"


def run_async_engine(path: str):
    if not os.path.exists(path):
        return

    config = RayInitConfig.from_yaml_path(path)
    engine_config = config.engine_config_data
    model = engine_config.model
    if model is None or not os.path.exists(model):
        return

    prompt = "Introduce some landmarks in London.\n The Tower of London is a historic castle on the north bank of the River Thames in central London. It was founded towards the end of 10"
    sampling_params = SamplingParams()
    asyncio.run(asy_for_loop_test(config, prompt, sampling_params))


async def get_result(engine, prompt, sampling_params):
    request_id = str(uuid.uuid4().hex)
    results = engine.generate(request_id, prompt, sampling_params)
    async for result in results:
        # print(result)
        assert result is not None


async def asy_for_loop_test(config, prompt, sampling_params):
    router_config = config.router_config_data
    engine_config = config.engine_config_data
    engine = Async_Engine(router_config=router_config, engine_config=engine_config)
    for i in range(10):
        print("in for loop", i)
        await get_result(engine, prompt, sampling_params)


def check_async_engine(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_async_engine(PATH)


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_async_engine():
    spawn(check_async_engine, 1)


if __name__ == "__main__":
    test_async_engine()
