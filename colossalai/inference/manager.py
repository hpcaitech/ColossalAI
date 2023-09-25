import time
import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from typing import Dict, List, Optional
from dynamic_batching.infer_batch import InferBatch
from ..sampling_params import SamplingParams
from inference.dynamic_batching import Req, Batch
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from dynamic_batching.req_queue import ReqQueue
from lightllm.utils.infer_utils import calculate_time
from dynamic_batching.io_struct import BatchTokenIdOut, AbortReq
from .stats import Stats

class DynamicBatchManager:

    def __init__(self,tp_engine, world_size, max_total_token_num, batch_max_tokens, running_max_req_size, eos_id, 
                 router_port, detokenization_port, model_rpc_ports, log_stats=True, log_stats_interval=10):
        self.engine = tp_engine
        self.world_size = world_size
        self.max_total_token_num = max_total_token_num

        self.req_queue = ReqQueue(max_total_token_num, batch_max_tokens, running_max_req_size) 
        # all the inputs should be put into req_queue 

        self.running_batch: Batch = None
        self.eos_id = eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10
        
        context = zmq.asyncio.Context(2)
        
        # self.send_to_detokenization = context.socket(zmq.PUSH)
        # self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")

        self.stats_tool = Stats(log_stats, log_stats_interval)

    # In Torch serve, model is initialized before manage
    async def wait_to_model_ready(self):
        pass

    def add_req(
        self,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str
    ):
        req = Req(request_id, prompt_ids, sampling_params)
        self.req_queue.append(req)
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        return

    async def abort(self, request_id):
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

    async def loop_for_fwd(self,):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    print("current batch size:", len(self.running_batch.reqs), "token used ratio:", self.running_batch.calcu_used_tokens() / self.max_total_token_num)
                    pass
                self.stats_tool.print_stats()
                
            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def _step(self):
        """
        handle the requests
        """
        # 删除所有已经 finished 的 req
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                await self._prefill_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        if self.has_wait_tokens < self.max_wait_tokens:
            self.stats_tool.count_output_tokens(self.running_batch)
            await self._decode_batch(self.running_batch)
            self._filter_runing_batch()
            self.has_wait_tokens += 1
            return
        else:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)
                await self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                self.has_wait_tokens = 0
            else:
                self.stats_tool.count_output_tokens(self.running_batch)
                await self._decode_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens += 1
        
        return

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        #rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        if self.world_size != 1:
            batch_id, reqs, dtype = obtain(batch_id), obtain(reqs), obtain(dtype)
        import torch
        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        batch_data = InferBatch.init_batch(batch_id, reqs, dtype, torch.cuda.current_device(), self.engine.model.mem_manager, self.engine.model.vocab_size)
        self.cache[batch_id] = batch_data
        return

    async def _prefill_batch(self, batch):
        await self._init_batch(batch)
        # rets = [self.model_rpcs[tp_rank].foward(batch.batch_id) for tp_rank in range(self.world_size)]
        # TODO: figure out if cache and batch id is needed
        rets = self.engine.prefill(batch.batch_id)
        ans = await asyncio.gather(*rets)

        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        # self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        return

    async def _decode_batch(self, batch:Batch):
        # rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        rets = self.engine.decode(batch.batch_id)
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0]) # gather or something
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        #self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        return

    async def _filter_batch(self, batch: Batch):
        req_id_list = [r.request_id for r in batch.reqs]
        filter_batch = batch.filter(req_id_list)
        batch = filter_batch
        # rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, req_id_list) for tp_rank in range(self.world_size)]
        # await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        # rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        # await asyncio.gather(*rets)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch2
        batch1 = m_batch    
        return

    async def _remove_batch(self, batch):
        batch.free_self()
        del batch
        # rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        # await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, has_new_finished_req):
        if has_new_finished_req:
            batch.filter_finished()
            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch)
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
        
    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 3:
                prompt_ids, sampling_params, request_id = recv_req
                self.add_req(prompt_ids, sampling_params, request_id)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
       #this logic should be implemented
       pass

def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):
    try:
        batch_manager = DynamicBatchManager(
            args.model_dir,
            world_size=args.tp,
            max_total_token_num=args.max_total_token_num,
            batch_max_tokens=args.batch_max_tokens,
            running_max_req_size=args.running_max_req_size,
            eos_id=args.eos_id,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            mode=mode,
            log_stats = not args.disable_log_stats,
            log_stats_interval = args.log_stats_interval)
    
        asyncio.run(batch_manager.wait_to_model_ready())
    except Exception as e:
        import traceback
        err_str = '\n'.join(traceback.format_exception(e))
        pipe_writer.send(err_str)
        batch_manager.clean_up()
        raise

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(batch_manager.loop_for_fwd())
    loop.run_until_complete(batch_manager.loop_for_netio_req())
    return
