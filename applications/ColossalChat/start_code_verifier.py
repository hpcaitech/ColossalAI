import os
from typing import List, Optional

from coati.distributed.reward.code_reward.utils import check_correctness  # Assuming utils.py is in the same directory
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

_API_KEY = os.environ.get("CODE_VERIFIER_API_KEY", "")
_MAX_TIMEOUT = 30


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
def check_correctness_api(request: CheckCorrectnessRequest, x_api_key: str = Header(...)):
    if not _API_KEY or x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        result, metadata = check_correctness(
            in_outs=request.in_outs,
            generation=request.generation,
            timeout=min(request.timeout, _MAX_TIMEOUT),
            debug=request.debug,
            eval_mode=request.eval_mode,
        )
        return CheckCorrectnessResponse(result=result, metadata=metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
