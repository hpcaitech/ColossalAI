import asyncio
from logging import Logger
from typing import AsyncIterator, Dict, Iterable, List, Optional, Set, Tuple, Type

from colossalai.inference.core.engine import InferenceEngine


class AsyncEngineDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(task: asyncio.Task, request_tracker: "RequestTracker") -> None:
    msg = "Task finished unexpectedly. This should never happen! " "Please open an issue on Github."
    try:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            raise AsyncEngineDeadError(msg + " See stack trace above for the actual cause.") from exc
        raise AsyncEngineDeadError(msg)
    except Exception as exc:
        request_tracker.propagate_exception(exc)
        raise exc


class AsyncStream:
    """A stream of Output for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopIteration)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self):
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        elif isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[int] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream, dict]] = asyncio.Queue()
        self.new_requests_event = None

    def __contains__(self, item):
        return item in self._request_streams

    def init_event(self):
        self.new_requests_event = asyncio.Event()

    def propagate_exception(self, exc: Exception, request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
        else:
            for stream in self._request_streams.values():
                stream.put(exc)

    def process_request_output(self, request_output, *, verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                Logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)

    def add_request(self, request_id: str, **engine_add_request_kwargs) -> AsyncStream:
        """
        Add a request to be sent to the engine on the next background
        loop iteration.
        """
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {"request_id": request_id, **engine_add_request_kwargs}))

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            Logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        self.new_requests_event.clear()

        return new_requests, finished_requests

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

        logits = await self.model(
            batch,
            self.k_cahce,
            self.v_cache,
        )

        if self.inference_config.pad_input:
            logits = logits[:, -1, :]

        self.request_handler.search_tokens(self.generation_config, logits)

        finished_sequences = self.request_handler.update()

        return finished_sequences


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
        self._request_tracker = RequestTracker()

    @property
    def background_loop_status(self):
        return self.background_loop is not None and not self.background_loop.done()

    def start_background_loop(self):
        if self.background_loop_status:
            raise RuntimeError("Existing loop is running")

        self._background_loop_unshielded = asyncio.get_event_loop().create_task(self.run_engine_loop())
        # self._background_loop_unshielded.add_done_callback(
        #     partial(_raise_exception_on_finish,
        #             request_tracker=self._request_tracker))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, **kwargs):
        return self._engine_class(**kwargs)

    async def step(self):
        """
        Run engine to process requests

        Returns True if there are in-progress requests.
        """
        request_outputs = await self.engine.async_step()

        for request_output in request_outputs:
            self._request_tracker.process_request_output(request_output, verbose=self.log_requests)

        return len(request_outputs) > 0

    async def abort(self, request_ids: Iterable[int]):
        self.engine.abort(request_ids)

    async def run_engine_loop(self):
        processing_requests = False
        while True:
            if not processing_requests:
                await self.request_tracker.wait_for_new_requests()
            processing_requests = await self.step()
            await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: int,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
    ) -> AsyncStream:
        if not self.background_loop_status:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise RuntimeError("Background loop is not running.")
        stream = self._request_tracker.add_request(
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
        try:
            stream = await self.add_request(request_id, prompt, prompt_token_ids=prompt_token_ids)
            async for request_output in stream:
                yield request_output

        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self.abort_request(request_id)
            raise e

    def abort_request(self, request_id: str) -> None:
        """
        Abort a request from request tracker.
        """
        self._request_tracker.abort_request(request_id)
