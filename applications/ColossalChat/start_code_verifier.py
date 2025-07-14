from typing import List, Optional

from coati.distributed.reward.code_reward.utils import check_correctness  # Assuming utils.py is in the same directory
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


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
def check_correctness_api(request: CheckCorrectnessRequest):
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
