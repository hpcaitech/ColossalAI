import asyncio
from dataclasses import dataclass

import pytest

from colossalai.inference.core.async_engine import AsyncInferenceEngine


@dataclass
class MockSequence:
    request_id: int


class MockEngine:
    def __init__(self):
        self.step_calls = 0
        self.add_request_calls = 0
        self.abort_request_calls = 0
        self.request_id = None

    async def async_step(self):
        self.step_calls += 1
        return ([MockSequence(request_id=self.request_id)], True) if self.request_id else ([], False)

    def add_single_request(self, **kwargs):
        del kwargs
        self.add_request_calls += 1

    def generate(self, request_id):
        self.request_id = request_id

    def stop_generating(self):
        self.request_id = None

    def add_request(self, **kwargs):
        del kwargs  # Unused
        self.add_request_calls += 1

    def abort_request(self, request_id):
        del request_id  # Unused
        self.abort_request_calls += 1


class MockAsyncInferenceEngine(AsyncInferenceEngine):
    def _init_engine(self, *args, **kwargs):
        return MockEngine()


@pytest.mark.asyncio
async def test_new_requests_event():
    engine = MockAsyncInferenceEngine()
    engine.start_background_loop()
    await asyncio.sleep(0.01)
    assert engine.engine.step_calls == 0

    await engine.add_request(1, "", None)
    await asyncio.sleep(0.01)
    assert engine.engine.add_request_calls == 1
    assert engine.engine.step_calls == 1

    await engine.add_request(2, "", None)
    engine.engine.generate(2)
    await asyncio.sleep(0)
    assert engine.engine.add_request_calls == 2
    assert engine.engine.step_calls == 2
    await asyncio.sleep(0)
    assert engine.engine.step_calls == 3
    engine.engine.stop_generating()
    await asyncio.sleep(0)
    assert engine.engine.step_calls == 4
    await asyncio.sleep(0)
    assert engine.engine.step_calls == 4

    await engine.add_request(3, "", None)
    await asyncio.sleep(0.01)
    assert engine.engine.add_request_calls == 3
    assert engine.engine.step_calls == 5
    await asyncio.sleep(0.01)
    assert engine.engine.add_request_calls == 3
    assert engine.engine.step_calls == 5
