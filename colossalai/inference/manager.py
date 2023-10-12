import time
from typing import List
import asyncio

from .dynamic_batching.infer_batch import InferBatch
from .dynamic_batching.io_struct import Batch, Req
from .dynamic_batching.req_queue import ReqQueue
from .dynamic_batching.sampling_params import SamplingParams
from .dynamic_batching.stats import Stats
from .tensor_parallel import TPInferEngine

from transformers import AutoTokenizer
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"

class DynamicBatchManager:
    def __init__(
        self,
        tp_engine: TPInferEngine,
        max_total_token_num,
        batch_max_tokens,
        eos_id,
        log_stats=True,
        log_stats_interval=10,
        running_batch: Batch = None,
        waiting_req_list: List = [],
    ):
        """
        Args:   tp_engine : The tp engine that dynamic batch manager hold, defined before dynamic batch manager
                max_total_token_num : max_total_token_num for memory manager, default to: max batch size * (max input len + max output len)
                batch_max_tokens : max tokens of one batch, default to (max input + output len) * num_requests
                running_max_req_size : max request size of running batch, equals to MAX_BATCH_SIZE of tp engine
                eos_id : The end token of a seq
                log_stats : whether to log stats
                log_stats_interval : log stats interval
                running_batch : running batch
                waiting_req_list : list of waiting requests, initialized before dynamic batch manager
        """
        self.engine = tp_engine
        self.max_total_token_num = max_total_token_num
        running_max_req_size = self.engine.max_batch_size if self.engine is not None else 2
        self.req_queue = ReqQueue(max_total_token_num, batch_max_tokens, running_max_req_size, waiting_req_list)
        # all the inputs should be put into req_queue: waiting req list

        self.running_batch: Batch = running_batch
        self.eos_id = eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10
        
        self.stats_tool = Stats(log_stats, log_stats_interval)
        self.mem_usage_interval = log_stats_interval * 2

    def add_req(self, prompt_ids: List[int], sampling_params: SamplingParams, request_id: str):
        """
        Add new request to req queue, during initialization all requests are held in waiting list.
        """
        req = Req(request_id, prompt_ids, sampling_params)
        self.req_queue.append(req)
        return

    def add_input(self, request_id, sampling_params, input_ids):
        """
        Encode and Add new input to req queue. support one sequence input for now.
        """
        prompt_ids = self.tokenizer.encode(input_ids)
        prompt_len = len(prompt_ids)
        if prompt_len > self.engine.max_input_len:
            raise ValueError(
                f"the input prompt token len {prompt_len} is too long > {self.engine.max_input_len}"
            )
        sampling_params.stop_sentences_to_token_ids(self.tokenizer)
        self.add_req(prompt_ids, sampling_params, request_id)
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

    async def loop_for_fwd(self):
        """
        The main loop for a dynamic batching process.
        """
        counter_count = 0
        #self.running_batch is not None or self.req_queue.waiting_req_list
        while True:
            async for item in self._step():
                yield item
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % self.mem_usage_interval == 0:
                    print(
                        "current batch size:",
                        len(self.running_batch.reqs),
                        "token used ratio:",
                        self.running_batch.calcu_used_tokens() / self.max_total_token_num,
                    )
                self.stats_tool.print_stats()

            if self.running_batch is None:
                time.sleep(0.1)  # 10ms

    def _set_tokenizer(self, tokenizer=None, tokenizer_name: str = "", trust_remote_code: bool = False, use_fast:bool = True,):
        if tokenizer is not None:
            self.tokenizer = tokenizer 
        else:
            if "llama" in tokenizer_name.lower() and use_fast == True:
                print(
                "For some LLaMA-based models, initializing the fast tokenizer may "
                "take a long time. To eliminate the initialization time, consider "
                f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
                "tokenizer. This is done automatically in Colossalai.")
                
                tokenizer_name = _FAST_LLAMA_TOKENIZER  
        
            try: 
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast,trust_remote_code=trust_remote_code)
            except TypeError as e:
                use_fast = False
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast,trust_remote_code=trust_remote_code)


    def _step(self):
        """
        Logic for handling requests
        """

        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                yield from self._prefill_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        if self.has_wait_tokens < self.max_wait_tokens:
            self.stats_tool.count_output_tokens(self.running_batch)
            yield from self._decode_batch(self.running_batch)
            self._filter_runing_batch()
            self.has_wait_tokens += 1
            return
        else:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)
                yield from self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                self.has_wait_tokens = 0
                
            else:
                self.stats_tool.count_output_tokens(self.running_batch)
                yield from self._decode_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens += 1
         
        return

    def _init_batch(self, batch: Batch, dtype="fp16"):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        batch_id = batch.batch_id

        import torch

        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"

        batch_data = InferBatch.init_batch(
            batch_id,
            reqs,
            dtype,
            torch.cuda.current_device(),
            self.engine.cache_manager,
            self.engine.model.config.vocab_size,
            self.engine.max_input_len + self.engine.max_output_len,
        )
        self.engine.cache[batch_id] = batch_data

    def _prefill_batch(self, batch):
        """
        For all batches, no matter it is a new batch or a mini batch, we need to do prefill first.
        """
        self._init_batch(batch)

        # TODO: figure out if cache and batch id is needed
        ans = self.engine._prefill_batch(batch.batch_id)
        req_to_out_token_id = ans
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        yield from self._handle_finish_req(batch, has_new_finished_req)
        
        # delete finished reqs

    def _decode_batch(self, batch: Batch):
        """
        Decoding process
        """
        ans = self.engine._decode_batch(batch.batch_id)
        req_to_out_token_id = ans
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        yield from self._handle_finish_req(batch, has_new_finished_req)

    def _filter_batch(self, batch: Batch):
        batch_id = batch.batch_id
        req_id_list = [r.request_id for r in batch.reqs]
        batch = self.engine.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list)
        del batch
        self.engine.cache[batch_id] = filter_batch

    def _merge_batch(self, batch1, batch2):
        """
        Merge new mini batch into running batch.
        """
        batch1 = self.engine.cache.pop(batch1.batch_id)
        batch2 = self.engine.cache.pop(batch2.batch_id)

        m_batch = InferBatch.merge(batch1, batch2)
        self.engine.cache[batch1.batch_id] = m_batch
        del batch1
        del batch2

    def _remove_batch(self, batch):
        """
        Remove finished batch.
        """
        batch = self.engine.cache.pop(batch.batch_id)
        batch.free_self()
        del batch

    def _handle_finish_req(self, batch: Batch, has_new_finished_req):
        if has_new_finished_req:
            finished_reqs=batch.filter_finished()
            if batch.is_clear():
                self._remove_batch(batch)
            else:
                self._filter_batch(batch)
            yield from self._output_process(finished_reqs)


    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None

    def _add_token_id_to_req(self, batch: Batch, req_ans):
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
        return

    async def _output_process(self, finished_reqs: List[Req]):
        """
        Process the output of a batch.
        """
        for req in finished_reqs:
            output = self.tokenizer.decode(req.output_ids)
            yield output, req.request_id, req.output_metadata_list 

    def clean_up(self):
        # this logic should be implemented in the future.
        pass

    async def generate(self,request_id,prompt_id,sampling_params):
        """
        Generate the output of a request.
        """
        self.add_input(request_id,prompt_id,sampling_params)
    

def start_dynamic_batching(args, tp_engine, waiting_req_list):
    try:
        batch_manager = DynamicBatchManager(
            tp_engine=tp_engine,
            max_total_token_num=args.max_total_token_num,
            batch_max_tokens=args.batch_max_tokens,
            eos_id=args.eos_id,
            log_stats=not args.disable_log_stats,
            log_stats_interval=args.log_stats_interval,
            waiting_req_list=waiting_req_list,
        )

    except Exception:
        batch_manager.clean_up()
        raise
    
    batch_manager._set_tokenizer(tokenizer_name = tp_engine.model.__class__.__name__)
    prod_task = asyncio.create_task(batch_manager.add_input(4,sampling_params=SamplingParams(),input_ids="hello world"))

    asyncio.run(prod_task)
    
    for item in batch_manager.loop_for_fwd():
        print(item)

    return batch_manager
