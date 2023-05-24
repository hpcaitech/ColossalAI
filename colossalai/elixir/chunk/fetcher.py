from typing import Optional

import torch

from .core import Chunk, ChunkGroup, TensorBlock, TensorState
from .scheduler import ChunkScheduler


class ChunkFetcher(object):

    def __init__(self,
                 scheduler: ChunkScheduler,
                 group: ChunkGroup,
                 overlap: bool = False,
                 reduce_always_fp32: bool = False) -> None:

        self.scheduler: ChunkScheduler = scheduler
        self.group: ChunkGroup = group
        self.reduce_always_fp32 = reduce_always_fp32
        self.current_step = -1

        self.overlap_flag = overlap
        self.main_stream = torch.cuda.current_stream()

        self.predict_next_chunk: Optional[Chunk] = None
        self.is_fetching: bool = False
        self.prefetch_stream = torch.cuda.Stream()

        self.reduced_chunk: Optional[Chunk] = None
        self.reduced_block: Optional[TensorBlock] = None
        self.reduce_stream = torch.cuda.Stream()

    def reset(self):
        self.scheduler.reset()
        self.current_step = -1

    def clear(self):
        if self.overlap_flag:
            torch.cuda.synchronize()
            self.predict_next_chunk = None
            self.is_fetching = False
            if self.reduced_chunk is not None:
                self.reduce_call_back()
                self.reduced_chunk = None
                self.reduced_block = None

        self.scheduler.clear()

    def trans_to_compute(self, tensors: list[torch.Tensor]):
        # update tensor states
        for t in tensors:
            self.group.tensor_trans_state(t, TensorState.COMPUTE)
        # chunk operations
        chunks = self.group.tensors_to_chunks(tensors)
        for chunk in chunks:
            self.scheduler.remove(chunk)
        return chunks

    def trans_to_hold(self, tensors: list[torch.Tensor], phase: str):
        assert phase in ('f', 'b')
        next_state = TensorState.HOLD if phase == 'f' else TensorState.HOLD_AFTER_BWD
        # update tensor states
        for t in tensors:
            self.group.tensor_trans_state(t, next_state)
        # chunk operations
        chunks = self.group.tensors_to_chunks(tensors)
        for chunk in chunks:
            if chunk.scatter_check:
                self.scheduler.add(chunk)

    def get_one_chunk(self, tensor: torch.Tensor) -> Chunk:
        return self.group.ten_to_chunk.get(tensor)

    def get_chunks(self, tensors: list[torch.Tensor]) -> list[Chunk]:
        return self.group.tensors_to_chunks(tensors)

    def is_in_fused(self, tensor: torch.Tensor):
        chunk = self.get_one_chunk(tensor)
        return chunk.rcache_fused

    def filter_chunks(self, chunks: list[Chunk]):
        return list(filter(lambda c: not self.group.is_accessed(c), chunks))

    def fetch_chunks(self, chunks: list[Chunk]):
        # make step + 1
        self.step()

        predict_hit = False
        # try to prefetch the next chunk
        if self.predict_next_chunk is not None and self.predict_next_chunk in chunks:
            if self.is_fetching:
                # prefetch hit, wait async prefetch
                self.main_stream.wait_stream(self.prefetch_stream)
                # clear prefetch information
                self.predict_next_chunk = None
                self.is_fetching = False

            predict_hit = True
        # filter accessed chunks
        scattered = self.filter_chunks(chunks)
        # sanity check: upload should wait for prefetch
        if self.predict_next_chunk is not None:
            assert len(scattered) == 0
        # all chunks are on the rcache
        if len(scattered) == 0:
            # prefetch if there is a hit above
            if predict_hit:
                self.prefetch(chunks)
            return

        for chunk in scattered:
            # if the rcache is not enough, just release a chunk
            if not self.group.rcache_enough_check(chunk):
                maybe_chunk = self.scheduler.top()
                # print(f'Evicting {chunk.chunk_id} -> {maybe_chunk.chunk_id}')
                if maybe_chunk is None:
                    raise RuntimeError('R cache is not enough. Try to allocate more.')
                self.scheduler.remove(maybe_chunk)
                self.group.release_chunk(maybe_chunk)

            # print('Accessing', chunk.chunk_id)
            self.group.access_chunk(chunk)

        if self.overlap_flag:
            assert self.predict_next_chunk is None
            self.prefetch(chunks)

    def reduce_call_back(self):
        self.reduced_chunk.update_extra_reduce_info(self.reduced_block)
        if self.reduced_block is not None:
            self.group.rcache.free_public_block(self.reduced_block)

    def reduce_chunk(self, chunk: Chunk):
        if not chunk.reduce_check:
            return False

        self.scheduler.remove(chunk)

        if not self.overlap_flag:
            # reduce the chunk if not overlapped
            self.group.reduce_chunk(chunk, always_fp32=self.reduce_always_fp32, sync=True)
        else:
            # wait main stream for its computation
            self.reduce_stream.wait_stream(self.main_stream)
            # main stream should wait reduce stream
            # if there is a block recycle
            if self.reduced_chunk is not None:
                self.main_stream.wait_stream(self.reduce_stream)
                self.reduce_call_back()

            with torch.cuda.stream(self.reduce_stream):
                self.reduced_chunk = chunk
                self.reduced_block = self.group.reduce_chunk(chunk, always_fp32=self.reduce_always_fp32, sync=False)

    def prefetch(self, chunks: list[Chunk]):
        next_chunk = self.scheduler.get_next_chunk(chunks)
        self.predict_next_chunk = next_chunk

        # return if there is no next scattered chunk
        if next_chunk is None or self.group.is_accessed(next_chunk):
            return

        evict_chunk = None
        if not self.group.rcache_enough_check(next_chunk):
            maybe_chunk = self.scheduler.top()
            # if there is no chunk can be evicted, just return
            if maybe_chunk is None:
                return
            # otherwise, release this chunk
            self.scheduler.remove(maybe_chunk)
            evict_chunk = maybe_chunk

        with torch.cuda.stream(self.prefetch_stream):
            # wait main stream
            self.prefetch_stream.wait_stream(self.main_stream)
            self.is_fetching = True

            if evict_chunk is not None:
                self.group.release_chunk(evict_chunk)
            self.group.access_chunk(next_chunk)

    def step(self):
        self.scheduler.step()
        self.current_step += 1
