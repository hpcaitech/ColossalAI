from typing import List

from .dynamic_batching.io_struct import Batch, Req, RequestOutput
from .manager import DynamicBatchManager
from .tensor_parallel import TPInferEngine


class Async_DynamicBatchManager(DynamicBatchManager):
    def __init__(
        self,
        tp_engine: TPInferEngine,
        max_total_token_num: int,
        batch_max_tokens: int,
        model: str,
        tokenizer=None,
        eos_id=None,
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
                model: the model weight dir path, the app will load config, weights and tokenizer from this dir
                log_stats : whether to log stats
                log_stats_interval : log stats interval
                running_batch : running batch
                waiting_req_list : list of waiting requests, initialized before dynamic batch manager
        """
        super().__init__(
            tp_engine,
            max_total_token_num,
            batch_max_tokens,
            model,
            tokenizer,
            eos_id,
            log_stats,
            log_stats_interval,
            running_batch,
            waiting_req_list,
        )

    def _step(self):
        """
        Logic for handling requests
        """
        has_new_finished = False
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                has_new_finished, outputs = self._prefill_batch(self.running_batch)
                self._filter_running_batch()
                self.has_wait_tokens = 0

        else:
            if self.has_wait_tokens < self.max_wait_tokens:
                self.stats_tool.count_output_tokens(self.running_batch)
                has_new_finished, outputs = self._decode_batch(self.running_batch)
                self._filter_running_batch()
                self.has_wait_tokens += 1

            else:
                new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
                if new_mini_batch is not None:
                    self.stats_tool.count_prompt_tokens(new_mini_batch)
                    has_new_finished, outputs = self._prefill_batch(new_mini_batch)
                    if not new_mini_batch.is_clear():
                        self._merge_batch(self.running_batch, new_mini_batch)
                        self.running_batch.merge(new_mini_batch)
                    self.has_wait_tokens = 0

                else:
                    self.stats_tool.count_output_tokens(self.running_batch)
                    has_new_finished, outputs = self._decode_batch(self.running_batch)
                    self._filter_running_batch()
                    self.has_wait_tokens += 1

        if has_new_finished:
            return outputs
        return None

    def _prefill_batch(self, batch):
        """
        For all batches, no matter it is a new batch or a mini batch, we need to do prefill first.
        """
        self._init_batch(batch)

        # TODO: figure out if cache and batch id is needed
        ans = self.engine._prefill_batch(batch.batch_id)
        req_to_out_token_id = ans
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id, self.engine.max_output_len)
        outputs = self._handle_finish_req(batch, has_new_finished_req)
        return has_new_finished_req, outputs
        # delete finished reqs

    def _decode_batch(self, batch: Batch):
        """
        Decoding process
        """
        ans = self.engine._decode_batch(batch.batch_id)
        req_to_out_token_id = ans
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id, self.engine.max_output_len)
        outputs = self._handle_finish_req(batch, has_new_finished_req)
        return has_new_finished_req, outputs

    def _handle_finish_req(self, batch: Batch, has_new_finished_req):
        if has_new_finished_req:
            finished_reqs = batch.filter_finished()
            if batch.is_clear():
                self._remove_batch(batch)
            else:
                self._filter_batch(batch)
            return self._output_process(finished_reqs)
        return None

    def _output_process(self, finished_reqs: List[Req]):
        """
        Process the output of a batch.
        """
        outputs = []
        for req in finished_reqs:
            output = self.tokenizer.decode(req.output_ids)
            outputs.append(RequestOutput(req.request_id, req.prompts, req.prompt_ids, output))
        return outputs


def start_dynamic_batching(args, tp_engine, waiting_req_list):
    try:
        batch_manager = Async_DynamicBatchManager(
            tp_engine=tp_engine,
            max_total_token_num=args.max_total_token_num,
            batch_max_tokens=args.batch_max_tokens,
            eos_id=args.eos_id,
            model=args.model,
            log_stats=not args.disable_log_stats,
            log_stats_interval=args.log_stats_interval,
            waiting_req_list=waiting_req_list,
        )

    except Exception:
        raise Exception

    return batch_manager
