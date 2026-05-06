import os
import secrets
from typing import List, Optional

from coati.distributed.reward.code_reward.utils import check_correctness  # Assuming utils.py is in the same directory
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

app = FastAPI()

API_KEY = os.environ.get("CODE_VERIFIER_API_KEY", "")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


class CheckCorrectnessRequest(BaseModel):
    in_outs: Optional[dict]
    generation: str
    timeout: int = 10
    debug: bool = True
    eval_mode: bool = False


class CheckCorrectnessResponse(BaseModel):
    result: List[int]
    metadata: List[dict]


@app.post("/check_correctness", response_model=CheckCorrectnessResponse)
def check_correctness_api(request: CheckCorrectnessRequest, _: str = Depends(verify_api_key)):
    try:
        result, metadata = check_correctness(
            in_outs=request.in_outs,
            generation=request.generation,
            timeout=request.timeout,
            debug=request.debug,
            eval_mode=request.eval_mode,
        )
        return CheckCorrectnessResponse(result=result, metadata=metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
