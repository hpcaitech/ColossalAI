# inspired by vLLM
import subprocess
import sys
import time

import pytest
import ray
import requests

MAX_WAITING_TIME = 300

pytestmark = pytest.mark.asyncio


@ray.remote(num_gpus=1)
class ServerRunner:
    def __init__(self, args):
        self.proc = subprocess.Popen(
            ["python3", "-m", "colossalai.inference.server.api_server"] + args,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_server()

    def ready(self):
        return True

    def _wait_for_server(self):
        # run health check
        start = time.time()
        while True:
            try:
                if requests.get("http://localhost:8000/v0/models").status_code == 200:
                    break
            except Exception as err:
                if self.proc.poll() is not None:
                    raise RuntimeError("Server exited unexpectedly.") from err

                time.sleep(0.5)
                if time.time() - start > MAX_WAITING_TIME:
                    raise RuntimeError("Server failed to start in time.") from err

    def __del__(self):
        if hasattr(self, "proc"):
            self.proc.terminate()


@pytest.fixture(scope="session")
def server():
    ray.init()
    server_runner = ServerRunner.remote(
        [
            "--model",
            "/home/chenjianghai/data/llama-7b-hf",
        ]
    )
    ray.get(server_runner.ready.remote())
    yield server_runner
    ray.shutdown()


async def test_completion(server):
    data = {"prompt": "How are you?", "stream": "False"}
    response = await server.post("v1/completion", json=data)
    assert response is not None


async def test_chat(server):
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "what is 1+1?"},
    ]
    data = {"messages": messages, "stream": "False"}
    response = await server.post("v1/chat", data)
    assert response is not None


if __name__ == "__main__":
    pytest.main([__file__])
