"""
Utils for model inference
"""
from colossalai.kernel.triton.copy_kv_cache_dest import copy_kv_cache_to_dest


def copy_kv_to_mem_cache(layer_id, key_buffer, value_buffer, context_mem_index, mem_manager):
    copy_kv_cache_to_dest(key_buffer, context_mem_index, mem_manager.key_buffer[layer_id])
    copy_kv_cache_to_dest(value_buffer, context_mem_index, mem_manager.value_buffer[layer_id])
    return
