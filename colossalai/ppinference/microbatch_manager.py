from .inference_config import InferenceConfig

__all__ = 'MicroBatchManager'


class MicroBatchManager():

    def __init__(
        self,
        pp_inference_config: InferenceConfig,
    ):
        self.pp_inference_config = pp_inference_config
        self.mb_to_kvcache = self._init_kvcache()
        self.cur_mb = 0

    def step(self, present_kv=None):
        self._update_kvcahe(present_kv)
        self.cur_mb = self.next_mb

    def _init_kvcache(self):
        mb_to_kvcache = {i: () for i in range(self.pp_inference_config.pp_size)}
        return mb_to_kvcache

    def _update_kvcahe(self, present_kv):
        self.mb_to_kvcache[self.cur_mb] += (present_kv,)

        if self.mb_to_kvcache[self.cur_mb][0][0][-2] == self.pp_inference_config.target_length:
            self.mb_to_kvcache.pop(self.cur_mb)

    @property
    def is_done(self):
        return len(self.mb_to_kvcache) == 0

    @property
    def next_mb(self):
        if self.is_done:
            return None

        nxt_mb = (self.cur_mb + 1) % self.pp_inference_config.pp_size
        while nxt_mb % self.pp_inference_config.pp_size not in self.mb_to_kvcache:
            nxt_mb = (nxt_mb + 1) % self.pp_inference_config.pp_size
        return nxt_mb

    @property
    def cur_kvcache(self):
        return self.mb_to_kvcache[self.cur_mb]
