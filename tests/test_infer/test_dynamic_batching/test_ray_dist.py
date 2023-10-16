import os
import uuid
from colossalai.inference.dynamic_batching.ray_init_config import RayInitConfig
from colossalai.inference.dynamic_batching.ray_dist_init import Driver
from colossalai.inference.dynamic_batching.sampling_params import SamplingParams
import asyncio

def test_ray_dist(path: str):
    print(f"Using yaml file {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Invalid yaml file path {path}")
    config = RayInitConfig.from_yaml_path(path)
    router_config = config.router_config_data
    engine_config = config.engine_config_data
    model = engine_config.model
    if model is None or not os.path.exists(model):
        raise ValueError("Model path not provided or invalid path!")
    
    driver = Driver(router_config=router_config, engine_config=engine_config)
    prompt = 'Introduce some landmarks in Beijing'
    
    request_id = str(uuid.uuid4().hex)
    
    sampling_params = SamplingParams()
    
    async def get_result(request_id, prompt, sampling_params):
        return await driver.generate(request_id, prompt, sampling_params)
    
    for test_async in [True, False]:
        if test_async: 
            print("test_async: ", test_async)
            result = get_result(request_id, prompt, sampling_params)                       
            print("result: ", result)
        else:
            print("test_async: ", test_async)
            result = driver.generate(request_id, prompt, sampling_params)                          
            print("result: ", result)
    
    
if __name__ == "__main__":
    path = "config.yaml"
    test_ray_dist(path)