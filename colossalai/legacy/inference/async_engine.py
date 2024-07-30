import asyncio

from colossalai.inference.dynamic_batching.ray_dist_init import Driver

from .dynamic_batching.io_struct import RequestOutput
from .dynamic_batching.sampling_params import SamplingParams


class RequestTracker:
    """
    A class for trace down all the requests, abstraction for async
    """

    def __init__(self) -> None:
        self._requests: asyncio.Queue[str] = asyncio.Queue()
        self._finished_requests: asyncio.Queue[RequestOutput] = asyncio.Queue()
        self.new_requests_event = None

    def __contains__(self, item):
        return item in self._requests

    def init_event(self):
        self.new_requests_event = asyncio.Event()

    def add_request(self, request_id: str):
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        self._requests.put_nowait(request_id)
        self.new_requests_event.set()  # NOTE: we may find a better way to clear this event

    def add_stop(self):
        """
        Add a StopIteration flag to stop async generator.
        """
        self._finished_requests.put_nowait(StopIteration)
        self.new_requests_event.clear()

    def process_request_output(self, request_output: RequestOutput) -> None:
        """Process a request output from the engine."""
        self._finished_requests.put_nowait(request_output)

    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._finished_requests.get()
        # print("result of ", result)
        if result is StopIteration:
            raise StopAsyncIteration
        return result


class Async_Engine:
    """
    Use an engine to launch RAY Driver --> RAY Worker --> Async_Manager
    Background loop: inference reqs in waiting list (Listen)
    Request Tracker: manage incoming requests and restore finished ones
    Generate: exposed func for add new input and return finished ones
    """

    def __init__(
        self,
        router_config,
        engine_config,
        start_engine_loop: bool = True,
    ) -> None:
        self.driver = Driver(router_config=router_config, engine_config=engine_config)
        self.background_loop = None
        self.start_engine_loop = start_engine_loop
        self._request_tracker = RequestTracker()

    def _step(self):
        """
        Logic for handling requests
        """
        request_outputs = self.driver.step()
        if request_outputs is not None:
            for request_output in request_outputs:
                self._request_tracker.process_request_output(request_output)
            self._request_tracker.add_stop()

    def abort_request(self, request_id: str):
        self.driver.abort(request_id)

    def _has_requests_in_progress(self):
        return self.driver.is_running()

    async def run_loop_fwd(self):
        has_requests_in_progress = self._has_requests_in_progress()
        while True:
            if not has_requests_in_progress:
                await self._request_tracker.wait_for_new_requests()
            self._step()
            await asyncio.sleep(0)

    @property
    def is_running(self):
        return self.background_loop is not None and not self.background_loop.done()

    def start_background_loop(self):
        if self.is_running:
            raise RuntimeError("Background loop is already running.")

        self._request_tracker.init_event()

        self.background_loop_unshielded = asyncio.get_event_loop().create_task(self.run_loop_fwd())
        self.background_loop = asyncio.shield(self.background_loop_unshielded)

    async def add_request(self, request_id: str, prompt: str, sampling_params: SamplingParams):
        self.driver.add_input(request_id, prompt, sampling_params)
        self._request_tracker.add_request(request_id)

    async def generate(self, request_id: str, prompt: str, sampling_params: SamplingParams):
        """
        The only exposed func, adding new request and return a async generator that yields the existing results.
        """
        try:
            if not self.is_running:
                self.start_background_loop()

            await self.add_request(request_id, prompt, sampling_params)

            async for request_output in self._request_tracker:
                yield request_output

        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the request.
            self.abort_request(request_id)
            raise e
