import asyncio
import logging
from functools import partial
from typing import AsyncIterator, Dict, Iterable, List, Optional, Tuple, Type

from colossalai.inference.core.engine import InferenceEngine

# CLI logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("colossalai-inference")


def _raise_exception_on_finish(task: asyncio.Task, request_tracker: "RequestTracker") -> None:
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
    """A stream of Output for a request that can be
    iterated over asynchronously."""

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
            return

        self._request_streams[request_id].set_result(None)

    def get_new_requests(self):
        """
        Get new requests from http server.
        """
        new_requests: List[Dict] = []

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        self.new_requests_event.clear()

        return new_requests

    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()


class _AsyncInferenceEngine(InferenceEngine):
    """
    Async methods for Inference Engine.
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
        loop = asyncio.get_running_loop()

        # Use run_in_executor to asyncally run the sync method model.forward().
        logits = await loop.run_in_executor(
            None,
            self.model,
            batch,
            self.k_cache,
            self.v_cache,
        )

        if self.inference_config.pad_input:
            logits = logits[:, -1, :]
        self.request_handler.search_tokens(self.generation_config, logits)
        # Return: List[Sequence]
        finished_sequences = self.request_handler.update()
        for sequence in finished_sequences:
            sequence.output = self.tokenizer.decode(sequence.output_token_id)

        return finished_sequences, self.request_handler.current_requests_in_batch() > 0


class AsyncInferenceEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the InferenceEngine class to make it asynchronous.
    It uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there are
    requests in the waiting queue. The generate method yields the outputs
    from the InferenceEngine to the caller.
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
    ) -> RequstStream:
        """
        Add a request to the background tracker(waitting queue), start the background loop if needed.
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
        )
        return stream

    async def generate(
        self,
        request_id: int,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate output from a request. It receives the request from http server, adds it into the
        waitting queue of Async Engine and streams the output sequence.

        """
        try:
            stream = await self.add_request(request_id, prompt, prompt_token_ids=prompt_token_ids)
            return await stream.get_result()

        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e
