import pytest
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

import colossalai
from colossalai.inference.dynamic_batching.io_struct import Req
from colossalai.inference.dynamic_batching.sampling_params import SamplingParams
from colossalai.inference.manager import DynamicBatchManager
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.shardformer import ShardConfig
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

TP_SIZE = 1
BATCH_SIZE = 2
MAX_INPUT_LEN = 48
MAX_OUTPUT_LEN = 256


def run():
    sampling_params = SamplingParams()

    req1 = Req(0, [1], sampling_params)
    req2 = Req(1, [2], sampling_params)
    req3 = Req(2, [3], sampling_params)
    # req 1-3 are initiliazed as token forward requests
    req4 = Req(3, [10, 10, 10, 9, 1], sampling_params)
    waiting_list = []
    waiting_list.append(req1)
    waiting_list.append(req2)
    waiting_list.append(req3)

    # init model and tp engine
    llama_config = LlamaConfig(num_hidden_layers=2, bos_token_id=0, eos_token_id=1, vocab_size=1200, hidden_size=1024)
    model = LlamaForCausalLM(llama_config)
    model = model.half()

    shard_config = ShardConfig(enable_tensor_parallelism=False, inference_only=True)
    infer_engine = TPInferEngine(model, shard_config, BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)

    dynamic_batch_manager = DynamicBatchManager(
        tp_engine=infer_engine,
        max_total_token_num=640,
        batch_max_tokens=608,
        eos_id=0,
        log_stats=False,
        log_stats_interval=10,
        waiting_req_list=waiting_list,
        model="llama",
    )
    before_add = len(dynamic_batch_manager.req_queue)

    # test add req function
    dynamic_batch_manager.add_req(req4.request_id, req4.prompt_ids, req4.sample_params)
    assert len(dynamic_batch_manager.req_queue.waiting_req_list) == before_add + 1

    # test abort function
    dynamic_batch_manager.abort(req4.request_id)
    assert dynamic_batch_manager.req_queue.waiting_req_list[-1].aborted == True

    # test filter batch function,  loop_for_fwd, _step, _init_batch and _prefill/_decode batch are tested
    batch = dynamic_batch_manager.req_queue.generate_new_batch()
    assert len(batch) == 2

    dynamic_batch_manager._init_batch(batch)
    assert dynamic_batch_manager.engine.cache[batch.batch_id] is not None

    batch.reqs[0].has_generate_finished = True
    # filter one finished
    batch.filter_finished()
    dynamic_batch_manager._filter_batch(batch)
    assert len(dynamic_batch_manager.engine.cache) == 1

    # test merge batch
    new_batch = dynamic_batch_manager.req_queue.generate_new_batch(batch)
    assert len(new_batch) == 1
    dynamic_batch_manager._init_batch(new_batch)
    dynamic_batch_manager._merge_batch(batch, new_batch)

    assert len(dynamic_batch_manager.engine.cache[batch.batch_id]) == 2


def check_dynamic_batching_manager(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_dynamic_batching_manager():
    spawn(check_dynamic_batching_manager, 1)


if __name__ == "__main__":
    test_dynamic_batching_manager()
