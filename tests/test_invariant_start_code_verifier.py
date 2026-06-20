import pytest
import importlib.util
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Load the actual production app from the file under test
spec = importlib.util.spec_from_file_location(
    "start_code_verifier",
    Path(__file__).parent / "applications/ColossalChat/start_code_verifier.py"
)
module = importlib.util.load_from_spec(spec)
spec.loader.exec_module(module)
app = module.app

client = TestClient(app, raise_server_exceptions=False)

VALID_PAYLOAD = {"generation": "def f(x): return x", "in_outs": "{}", "timeout": 5, "debug": False, "eval_mode": "exec"}

@pytest.mark.parametrize("headers,payload", [
    # Exact exploit: no auth, arbitrary code execution attempt
    ({}, {"generation": "__import__('os').system('id')", "in_outs": "{}", "timeout": 10, "debug": False, "eval_mode": "exec"}),
    # Missing token (boundary: empty Authorization header)
    ({"Authorization": ""}, VALID_PAYLOAD),
    # Malformed token
    ({"Authorization": "Bearer not.a.valid.jwt"}, VALID_PAYLOAD),
    # Expired/fake token
    ({"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjF9.invalid"}, VALID_PAYLOAD),
])
def test_check_correctness_requires_authentication(headers, payload):
    """Invariant: /check_correctness must reject unauthenticated or invalidly-authenticated requests with 401 or 403."""
    response = client.post("/check_correctness", json=payload, headers=headers)
    assert response.status_code in (401, 403), (
        f"Expected 401 or 403 for unauthenticated request, got {response.status_code}. "
        "The endpoint must not be publicly accessible without valid credentials."
    )