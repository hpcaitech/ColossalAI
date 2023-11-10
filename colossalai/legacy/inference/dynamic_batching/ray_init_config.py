import logging

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EngineArgsClass(BaseModel):
    """Config for Engine"""

    model: str
    tensor_parallel_size: int = 2
    max_batch_size: int = 4
    max_input_len: int = 128
    max_output_len: int = 32


class RooterArgsClass(BaseModel):
    """Config for Rooter"""

    max_total_token_num: int = 42
    batch_max_tokens: int = 42
    eos_id: int = 0
    disable_log_stats: bool = False
    log_stats_interval: int = 10
    model: str


class RayInitConfig(BaseModel):
    """All-together configs without app router config"""

    engine_config_data: EngineArgsClass
    router_config_data: RooterArgsClass

    @classmethod
    def from_yaml_path(cls, path: str):
        try:
            with open(path, "r") as yaml_file:
                try:
                    config = yaml.safe_load(yaml_file)
                    # serve deployment config
                    engine_config = config.get("engine_config", {})
                    router_config = config.get("router_config", {})

                    return cls(
                        engine_config_data=engine_config,
                        router_config_data=router_config,
                    )
                except yaml.YAMLError as e:
                    logger.error(f"An Error occurred when parsing yaml: {e}")
                    raise
        except FileNotFoundError:
            logger.error(f"The file '{path}' does not exist!")
            raise
        except OSError as e:
            logger.error(f"An Error occurred: {e}")
            raise
