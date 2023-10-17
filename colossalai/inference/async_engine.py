import asyncio

from colossalai.inference.dynamic_batching.ray_dist_init import Driver

from .dynamic_batching.sampling_params import SamplingParams


class RequestTracker:
    """
    A class for trace down all the requests, abstraction for async
    """

    def __init__(self) -> None:
        self._requests: asyncio.Queue[str] = asyncio.Queue()
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self.new_requests_event = None

    def __contains__(self, item):
        return item in self._requests

    def init_event(self):
        self.new_requests_event = asyncio.Event()

    def add_request(self, request_id: str):
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        # if request_id in self._requests:
        #     raise KeyError(f"Request {request_id} already exists.")

        self._requests.put_nowait(request_id)

        self.new_requests_event.set()

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)
        return

    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()


class Async_Engine:

    """
    loop: start listen
    add req
    remove req
    generate--> return async generator
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
        self.driver.step()

    def _has_requests_in_progress(self):
        return self.driver.has_requests_in_progress()

    async def run_loop_fwd(self):
        has_requests_in_progress = self._has_requests_in_progress()
        while True:
            if not has_requests_in_progress:
                await self._request_tracker.wait_for_requests()
            has_requests_in_progress = await self._step()
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

            stream = await self.driver.async_generate(request_id, prompt, sampling_params)

            async for request_output in stream:
                yield request_output

        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the request.
            self._request_tracker.abort_request(request_id)
            raise e


#     def generate(self, request_id: str, prompt: str, sampling_params: SamplingParams):
#         results = ray.get([w.generate.remote(request_id, prompt, sampling_params) for w in self.workers])
#         text_res = results[0]  # get any one of the copies
#         return text_res

#     async def async_generate(self, request_id: str, prompt: str, sampling_params: SamplingParams):
#         all_outputs = []
#         for worker in self.workers:
#             all_outputs.append(worker.generate.remote(request_id, prompt, sampling_params))
#         all_outputs = await asyncio.gather(*all_outputs)
#         text_res = all_outputs[0]# get any one of the copies
#         return text_res

#     def add_input(self, request_id: str, prompt: str, sampling_params: SamplingParams):
#         ray.get([w.add_input.remote(request_id, sampling_params, prompt) for w in self.workers])

#     def abort(self,request_id: str):
#         ray.get([w.abort.remote(request_id) for w in self.workers])

#     def step(self):
#         ray.get([w._step.remote() for w in self.workers])

#     def add_req(self, prompt_ids: List[int], sampling_params: SamplingParams, request_id: str, prompt: str):
#         ray.get([w.add_req.remote(prompt_ids, sampling_params, request_id, prompt) for w in self.workers])
