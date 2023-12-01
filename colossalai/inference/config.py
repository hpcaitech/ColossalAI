"""
Our config consists of three parts:
    1. model_config: The configuration for the model, including `model name`, 'model path' and self-defined layer.
    2. parallel_config: The configuration for parallelize model, including `tp_size`,'pp_size', `world size`, `local rank`, `master port`, `master ip`.
    3. cache_config: Configuration for initialize and manage kv cache, including `block size`, `block num`
For the convenience of users, we provide a unified config api for that wrapped all the configs. One can easily construct a colossal_config by setting the needed configs.
"""
