import argparse
import time
from typing import List

from dynamic_batching.infer_batch import InferBatch
from dynamic_batching.io_struct import Batch, Req
from dynamic_batching.req_queue import ReqQueue
from dynamic_batching.sampling_params import SamplingParams
from dynamic_batching.stas import Stats
from rpyc.utils.classic import obtain
from tensor_parallel.engine import TPInferEngine
from transformers import LlamaForCausalLM, LlamaTokenizer

import colossalai
from colossalai.shardformer import ShardConfig
from tests.test_infer.test_llama_infer import init_to_get_rotary


# faulthandler.enable()
class DynamicBatchManager:
    def __init__(
        self,
        tp_engine: TPInferEngine,
        world_size,
        max_total_token_num,
        batch_max_tokens,
        running_max_req_size,
        eos_id,
        log_stats=True,
        log_stats_interval=10,
        running_batch: Batch = None,
        waiting_req_list=[],
    ):
        self.engine = tp_engine
        self.world_size = world_size
        self.max_total_token_num = max_total_token_num

        self.req_queue = ReqQueue(max_total_token_num, batch_max_tokens, running_max_req_size, waiting_req_list)
        # all the inputs should be put into req_queue

        self.running_batch: Batch = running_batch
        self.eos_id = eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10

        # context = zmq.asyncio.Context(2)
        # self.send_to_detokenization = context.socket(zmq.PUSH)
        # self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")

        self.stats_tool = Stats(log_stats, log_stats_interval)

    # In Torch serve, model is initialized before manage
    def wait_to_model_ready(self):
        pass

    def add_req(self, prompt_ids: List[int], sampling_params: SamplingParams, request_id: str):
        req = Req(request_id, prompt_ids, sampling_params)
        self.req_queue.append(req)
        return

    def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True
        return

    def loop_for_fwd(self):
        counter_count = 0
        while self.running_batch is not None or self.req_queue.waiting_req_list:
            self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    print(
                        "current batch size:",
                        len(self.running_batch.reqs),
                        "token used ratio:",
                        self.running_batch.calcu_used_tokens() / self.max_total_token_num,
                    )
                self.stats_tool.print_stats()

            if self.running_batch is None:
                time.sleep(10)  # 10ms

    def _step(self):
        """
        handle the requests
        """
        # 删除所有已经 finished 的 req
        print("in step forward")
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                print(new_batch.reqs)
                self._prefill_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        if self.has_wait_tokens < self.max_wait_tokens:
            self.stats_tool.count_output_tokens(self.running_batch)
            self._decode_batch(self.running_batch)
            self._filter_runing_batch()
            self.has_wait_tokens += 1
            return
        else:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)
                self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                self.has_wait_tokens = 0
            else:
                self.stats_tool.count_output_tokens(self.running_batch)
                self._decode_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens += 1

        return

    def _init_batch(self, batch: Batch, dtype="fp16"):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        batch_id = batch.batch_id
        # rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        if self.world_size != 1:
            batch_id, reqs, dtype = obtain(batch_id), obtain(reqs), obtain(dtype)
        import torch

        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        # cache may be removed
        batch_data = InferBatch.init_batch(
            batch_id,
            reqs,
            dtype,
            torch.cuda.current_device(),
            self.engine.cache_manager,
            self.engine.model.config.vocab_size,
        )
        self.engine.cache[batch_id] = batch_data
        return

    def _prefill_batch(self, batch):
        self._init_batch(batch)
        # rets = [self.model_rpcs[tp_rank].foward(batch.batch_id) for tp_rank in range(self.world_size)]
        # TODO: figure out if cache and batch id is needed
        rets = self.engine._prefill_batch(batch.batch_id)
        ans = rets
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        # self._send_to_detokenization_proc(batch, req_to_out_token_id)
        self._handle_finish_req(batch, has_new_finished_req)
        return

    def _decode_batch(self, batch: Batch):
        # rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        rets = self.engine._decode_batch(batch.batch_id)
        ans = rets
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])  # gather or something
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        # self._send_to_detokenization_proc(batch, req_to_out_token_id)
        self._handle_finish_req(batch, has_new_finished_req)
        return

    def _filter_batch(self, batch: Batch):
        batch_id = batch.batch_id
        req_id_list = [r.request_id for r in batch.reqs]
        if self.world_size != 1:
            batch_id, req_id_list = obtain(batch_id), obtain(req_id_list)
        batch = self.engine.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list)
        del batch
        self.engine.cache[batch_id] = filter_batch
        return

    def _merge_batch(self, batch1, batch2):
        batch1 = self.engine.cache.pop(batch1.batch_id)
        batch2 = self.engine.cache.pop(batch2.batch_id)
        # rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        # await asyncio.gather(*rets)

        m_batch = InferBatch.merge(batch1, batch2)
        self.engine.cache[batch1.batch_id] = m_batch
        del batch1
        del batch2
        return

    def _remove_batch(self, batch):
        batch = self.engine.cache.pop(batch.batch_id)
        batch.free_self()
        del batch
        # rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        # await asyncio.gather(*rets)
        return

    def _handle_finish_req(self, batch: Batch, has_new_finished_req):
        if has_new_finished_req:
            batch.filter_finished()
            if batch.is_clear():
                self._remove_batch(batch)
            else:
                self._filter_batch(batch)
        return

    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return

    def _add_token_id_to_req(self, batch: Batch, req_ans):
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
        return

    def clean_up(self):
        # this logic should be implemented
        pass


def start_router_process(args, tp_engine, waiting_req_list):
    try:
        batch_manager = DynamicBatchManager(
            tp_engine=tp_engine,
            world_size=args.tp,
            max_total_token_num=args.max_total_token_num,
            batch_max_tokens=args.batch_max_tokens,
            running_max_req_size=args.running_max_req_size,
            eos_id=args.eos_id,
            log_stats=not args.disable_log_stats,
            log_stats_interval=args.log_stats_interval,
            waiting_req_list=waiting_req_list,
        )

    except Exception:
        # may need use logger
        batch_manager.clean_up()
        raise

    print("start router process")
    batch_manager.loop_for_fwd()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, help="tp_size", default=1)
    parser.add_argument("--max_total_token_num", type=int, default=42, help="max_total_token_num")
    parser.add_argument("-b", "--batch_max_tokens", type=int, default=42, help="max tokens of one batch")
    parser.add_argument("--running_max_req_size", type=int, default=2, help="max request size of running batch ")
    parser.add_argument("--eos_id", type=int, default=0, help="The end token of a seq")
    parser.add_argument("--disable_log_stats", type=bool, default=False)
    parser.add_argument("--log_stats_interval", type=int, default=10)
    args = parser.parse_args()
    sampling_params = SamplingParams()

    req1 = Req(0, [10, 10, 10, 10, 10], sampling_params)
    req2 = Req(1, [10, 10, 10, 10, 10], sampling_params)
    waiting_list = []
    waiting_list.append(req1)
    waiting_list.append(req2)

    colossalai.launch(config={}, rank=0, world_size=1, host="localhost", port=8081, backend="nccl")
    tokenizer = LlamaTokenizer.from_pretrained("/data/scratch/llama-7b-hf")
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model = LlamaForCausalLM.from_pretrained("/data/scratch/llama-7b-hf", pad_token_id=tokenizer.eos_token_id)
    model = model.half()
    init_to_get_rotary(model.model, base=10000)
    shard_config = ShardConfig(enable_tensor_parallelism=False, inference_only=True)
    infer_engine = TPInferEngine(model, shard_config, 2, 5, 16)
    start_router_process(args=args, tp_engine=infer_engine, waiting_req_list=waiting_list)
