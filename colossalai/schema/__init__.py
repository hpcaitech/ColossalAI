from .collated import ColossalaiConfig
from pydantic import ValidationError

__all__ = ['ColossalaiConfig', 'validate_config']


def validate_config(config: dict):
    # will raise validation error
    # if config does not match with the schema
    try:
        ColossalaiConfig(**config)
    except ValidationError as e:
        raise ValueError(f"Your configuration file failed to match the schema ({e}), "
                         "please visit https://www.colossalai.org to verify your configuration")
