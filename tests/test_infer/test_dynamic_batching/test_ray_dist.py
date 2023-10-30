import asyncio
import os
import uuid

import pytest

import colossalai
from colossalai.inference.dynamic_batching.ray_dist_init import Driver
from colossalai.inference.dynamic_batching.ray_init_config import RayInitConfig
from colossalai.inference.dynamic_batching.sampling_params import SamplingParams
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

PATH = "config.yaml"


def run_ray_dist(path: str):
    if not os.path.exists(path):
        return
    config = RayInitConfig.from_yaml_path(path)
    router_config = config.router_config_data
    engine_config = config.engine_config_data
    model = engine_config.model
    if model is None or not os.path.exists(model):
        return
    driver = Driver(router_config=router_config, engine_config=engine_config)
    prompt = "Introduce some landmarks in Beijing"

    request_id = str(uuid.uuid4().hex)
    sampling_params = SamplingParams()
    print("sampling_params: ", sampling_params)

    async def get_result(request_id, prompt, sampling_params):
        return await driver.async_generate(request_id, prompt, sampling_params)

    for test_async in [True, False]:
        if test_async:
            print("test_async: ", test_async)
            result = asyncio.run(get_result(request_id, prompt, sampling_params))
            assert result is not None
            print("result: ", result)
        else:
            print("test_async: ", test_async)
            result = driver.generate(request_id, prompt, sampling_params)
            assert result is not None
            print("result: ", result)

    is_running = None
    is_running = driver.is_running()
    assert is_running is not None
    print("is_running: ", is_running)


def check_ray_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_ray_dist(PATH)


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_ray_dist():
    spawn(check_ray_dist, 1)


if __name__ == "__main__":
    test_ray_dist()
