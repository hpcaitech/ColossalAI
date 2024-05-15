import asyncio
import logging
from functools import partial
from typing import AsyncIterator, Dict, Iterable, List, Optional, Set, Tuple, Type

from colossalai.inference.core.engine import InferenceEngine
from colossalai.inference.sampler import search_tokens

# CLI logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("colossalai-inference")


def _raise_exception_on_finish(task: asyncio.Task, request_tracker: "Tracer") -> None:
    msg = "Task finished unexpectedly. This should never happen! "
    try:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            raise RuntimeError(msg + " See stack trace above for the actual cause.") from exc
        raise RuntimeError(msg)
    except Exception as exc:
        request_tracker.propagate_exception(exc)
        raise exc


class RequstStream:
    """
    A stream of Output for a request that can be iterated over asynchronously.
        Attributes: 1.request_id: The id of the request.
                    2._future: A future that will be set when the request is finished.
        Methods: set_result and get_result, results will be set when finished, for once, and
        the `self.future` will be set to done.

    """

    def __init__(self, request_id: int) -> None:
        self.request_id = request_id
        self._future = asyncio.Future()

    def set_result(self, result) -> None:
        """Set final result and  signal taht it's ready"""
        if not self._future.done():
            self._future.set_result(result)

    async def get_result(self):
        """Wait for the result to be set and return it."""
        return await self._future

    @property
    def finished(self) -> bool:
        """Check if the stream has finished by checking if the future is done."""
        return self._future.done()


class Tracer:
    """
    Recording new requests and finished requests.
        Attributes: 1._request_streams: We create one stream for each request to trace the output.
                    2._finished_requests: A queue to store the finished requests.
                    3._new_requests: New requests will be stored in this queue first, before sending them to the engine.
                    4.new_requests_event: An event to notify the engine that there are new requests.
    """

    def __init__(self) -> None:
        self._request_streams: Dict[int, RequstStream] = {}
        self._finished_requests: asyncio.Queue[int] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[RequstStream, dict]] = asyncio.Queue()
        self.new_requests_event = None

    def __contains__(self, item):
        return item in self._request_streams

    def init_event(self):
        self.new_requests_event = asyncio.Event()

    def propagate_exception(self, exc: Exception, request_id: Optional[int] = None) -> None:
        """
        Propagate an exception to request streams (all if request_id is None).
        """
        if request_id is not None:
            self._request_streams[request_id].set_result(exc)
        else:
            for stream in self._request_streams.values():
                stream.set_result(exc)

    def process_finished_request(self, finished_request) -> None:
        """Process a finished request from the engine."""
        request_id = finished_request.request_id
        try:
            self._request_streams[request_id].set_result(finished_request)
        except:
            raise RuntimeError(f"The request_id {request_id} is not found in our stream, please check")
        self.abort_request(request_id)

    def add_request(self, request_id: int, **engine_add_request_kwargs) -> RequstStream:
        """
        Add a request to be sent to the engine on the next background
        loop iteration.
        """
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = RequstStream(request_id)
        logger.info(f"Added request {request_id}.")
        self._new_requests.put_nowait((stream, {"request_id": request_id, **engine_add_request_kwargs}))
        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: int, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[request_id].finished:
            # The request has already finished or been aborted.
            # The requests in new_requests will be aborted when try to get them(if marked aborted)
            return

        self._request_streams[request_id].set_result(None)

    def get_new_requests(self):
        """
        Get new requests from http server.
        """
        new_requests: List[Dict] = []
        finished_requests: Set[int] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if new_request["request_id"] in finished_requests:
                # The request has been aborted.
                stream.set_result(None)
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        self.new_requests_event.clear()

        return new_requests

    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()


class _AsyncInferenceEngine(InferenceEngine):
    """
    Async methods for Inference Engine. This engine is an extension for InferenceEngine, and the additional methods will only be used for
        Methods: 1. async_step: The async version of Engine.step()
    """

    async def async_step(self) -> List[str]:
        """
        The async version of Engine.step()
        Performs one decoding iteration and returns newly generated results.

        It first schedules the sequences to be executed in the next iteration.
        Then, it executes the model and updates the scheduler with the model
        outputs. Finally, it decodes the sequences and returns the newly
        generated results.
        """
        batch = self.request_handler.schedule()
        input_token_ids, output_tensor, input_meta_data = self.prepare_input(batch)

        loop = asyncio.get_running_loop()

        if input_meta_data.use_cuda_graph:
            model_executable = self.graph_runners[input_meta_data.batch_size]
        else:
            model_executable = self.model

        # Use run_in_executor to asyncally run the sync method model.forward().
        logits = await loop.run_in_executor(
            None,
            model_executable,
            input_token_ids,
            output_tensor,
            input_meta_data,
            self.k_cache,
            self.v_cache,
        )

        if self.inference_config.pad_input:
            logits = logits[:, -1, :]
        next_tokens = search_tokens(
            self.generation_config, logits, input_meta_data.is_prompts, batch_token_ids=input_meta_data.batch_token_ids
        )

        self.request_handler.append_next_tokens(next_tokens)
        finished_sequences = self.request_handler.update()

        for sequence in finished_sequences:
            sequence.output = self.tokenizer.decode(sequence.output_token_id)

        return finished_sequences, not self.request_handler.running_list.is_empty()

    def add_single_request(self, request_id: int, prompt: str, prompt_token_ids, generation_config=None):
        prompts = [prompt]
        gen_config_dict = generation_config.to_dict() if generation_config is not None else {}
        self.add_request(request_ids=request_id, prompts=prompts, prompts_token_ids=prompt_token_ids, **gen_config_dict)


class AsyncInferenceEngine:
    """An asynchronous wrapper for the InferenceEngine class.

    This class is used to wrap the InferenceEngine class to make it asynchronous.
    It uses asyncio to create a background loop that keeps processing incoming
    requests. Note that this class does not hold model directly, when incoming a new
    request, it first called `add_request` and the Tracer will record the request, putting
    it to the background `InferenceEngine`(done in background loop) to process. You can
    consider this engine as an interface.
    """

    _engine_class: Type[_AsyncInferenceEngine] = _AsyncInferenceEngine

    def __init__(self, start_engine_loop: bool = True, **kwargs):
        self.engine = self._init_engine(**kwargs)
        self.background_loop = None
        # reference to the unshielded loop
        self._background_loop_unshielded = None
        self.start_engine_loop = start_engine_loop
        self._request_tracer = Tracer()

    @property
    def background_loop_status(self):
        return self.background_loop is not None and not self.background_loop.done()

    def start_background_loop(self):
        if self.background_loop_status:
            raise RuntimeError("Existing loop is running")

        self._request_tracer.init_event()

        self._background_loop_unshielded = asyncio.get_event_loop().create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish, request_tracker=self._request_tracer)
        )
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, **kwargs):
        return self._engine_class(**kwargs)

    async def step(self):
        """
        Run engine to process requests

        Returns True if there are in-progress requests.
        """
        new_requests = self._request_tracer.get_new_requests()
        for new_request in new_requests:
            self.engine.add_single_request(**new_request)
        newly_finished_seqs, has_running_requests = await self.engine.async_step()
        for seq in newly_finished_seqs:
            self._request_tracer.process_finished_request(seq)

        return has_running_requests

    async def _engine_abort(self, request_ids: Iterable[int]):
        self.engine.abort_request(request_ids)

    async def abort(self, request_id: int):
        """
        Abort a single request
        """
        if not self.background_loop_status:
            raise RuntimeError("Background loop is not running or launched correctly.")
        return self._abort(request_id)

    def _abort(self, request_id: int):
        self._request_tracer.abort_request(request_id)

    async def run_engine_loop(self):
        processing_requests = False
        while True:
            if not processing_requests:
                await self._request_tracer.wait_for_new_requests()
            processing_requests = await self.step()
            await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: int,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        generation_config=None,
    ) -> RequstStream:
        """
        Add a request to the background tracker(waiting queue), start the background loop if needed.
        """
        if not self.background_loop_status:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise RuntimeError("Background loop is not running.")
        stream = self._request_tracer.add_request(
            request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            generation_config=generation_config,
        )
        return stream

    async def generate(
        self,
        request_id: int,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        generation_config=None,
    ) -> AsyncIterator[str]:
        """
        Generate output from a request. It receives the request from http server, adds it into the
        waitting queue of Async Engine and streams the output sequence.
        """
        try:
            stream = await self.add_request(
                request_id, prompt, prompt_token_ids=prompt_token_ids, generation_config=generation_config
            )
            return await stream.get_result()

        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the request.
            self._abort(request_id)
            raise e
