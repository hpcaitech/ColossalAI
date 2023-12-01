class RequestHandler:
    def __init__(self, cache_config) -> None:
        self.cache_config = cache_config
        self._init_cache()

    def _init_cache(self):
        pass

    def schedule(self, request):
        pass
