import argparse
import os
from threading import Lock
from typing import Generator, List, Optional

import torch
import uvicorn
from coati.models import generate_streaming
from coati.quant import llama_load_quant, low_resource_init
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils import ChatPromptProcessor, Dialogue, LockedIterator, load_json, update_model_kwargs_fn

MAX_LEN = 512
running_lock = Lock()


class GenerationTaskReq(BaseModel):
    max_new_tokens: int = Field(gt=0, le=512, example=64)
    history: List[Dialogue] = Field(min_items=1)
    top_k: Optional[int] = Field(default=None, gt=0, example=50)
    top_p: Optional[float] = Field(default=None, gt=0.0, lt=1.0, example=0.5)
    temperature: Optional[float] = Field(default=None, gt=0.0, lt=1.0, example=0.7)
    repetition_penalty: Optional[float] = Field(default=None, gt=1.0, example=1.2)


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# set CORS
origin_spec_from_env = os.environ.get("CORS_ORIGIN", None)

if origin_spec_from_env is not None:
    # allow CORS from the specified origins
    origins = os.environ["CORS_ORIGIN"].split(",")
else:
    # allow CORS from all origins
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_streamingly(prompt, max_length, max_new_tokens, top_k, top_p, temperature):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    # TODO(ver217): streaming generation does not support repetition_penalty now
    model_kwargs = {
        "max_new_tokens": max_new_tokens,
        "early_stopping": True,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "prepare_inputs_fn": None,
        "update_model_kwargs_fn": update_model_kwargs_fn,
    }
    is_first_word = True
    generator = LockedIterator(
        generate_streaming(model, input_ids, tokenizer, max_length, **model_kwargs), running_lock
    )
    for output in generator:
        output = output.cpu()
        tokens = tokenizer.convert_ids_to_tokens(output, skip_special_tokens=True)
        current_sub_tokens = []
        for token in tokens:
            if token in tokenizer.all_special_tokens:
                continue
            current_sub_tokens.append(token)
        if current_sub_tokens:
            out_string = tokenizer.sp_model.decode(current_sub_tokens)
            if is_first_word:
                out_string = out_string.lstrip()
                is_first_word = False
            elif current_sub_tokens[0].startswith("▁"):
                # whitespace will be ignored by the frontend
                out_string = " " + out_string
            yield out_string


async def event_generator(request: Request, generator: Generator):
    while True:
        if await request.is_disconnected():
            break
        try:
            yield {"event": "generate", "data": next(generator)}
        except StopIteration:
            yield {"event": "end", "data": ""}
            break


@app.post("/generate/stream")
@limiter.limit("1/second")
def generate(data: GenerationTaskReq, request: Request):
    prompt = prompt_processor.preprocess_prompt(data.history)
    event_source = event_generator(
        request,
        generate_streamingly(prompt, data.max_length, data.max_new_tokens, data.top_k, data.top_p, data.temperature),
    )
    return EventSourceResponse(event_source)


@app.post("/generate")
@limiter.limit("1/second")
def generate_no_stream(data: GenerationTaskReq, request: Request):
    prompt = prompt_processor.preprocess_prompt(data.history, data.max_new_tokens)
    if prompt_processor.has_censored_words(prompt):
        return prompt_processor.SAFE_RESPONSE
    inputs = {k: v.cuda() for k, v in tokenizer(prompt, return_tensors="pt").items()}
    with running_lock:
        output = model.generate(**inputs, **data.dict(exclude={"history"}))
    output = output.cpu()
    prompt_len = inputs["input_ids"].size(1)
    response = output[0, prompt_len:]
    out_string = tokenizer.decode(response, skip_special_tokens=True)
    out_string = prompt_processor.postprocess_output(out_string)
    if prompt_processor.has_censored_words(out_string):
        return prompt_processor.SAFE_RESPONSE
    return out_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pretrained",
        help="Path to pretrained model. Can be a local path or a model name from the HuggingFace model hub.",
    )
    parser.add_argument(
        "--tokenizer_path",
        help="Path to pretrained tokenizer. Can be a local path or a model name from the HuggingFace model hub.",
        default=None,
    )
    parser.add_argument(
        "--quant",
        choices=["8bit", "4bit"],
        default=None,
        help="Quantization mode. Default: None (no quantization, fp16).",
    )
    parser.add_argument(
        "--gptq_checkpoint",
        default=None,
        help="Path to GPTQ checkpoint. This is only useful when quantization mode is 4bit. Default: None.",
    )
    parser.add_argument(
        "--gptq_group_size",
        type=int,
        default=128,
        help="Group size for GPTQ. This is only useful when quantization mode is 4bit. Default: 128.",
    )
    parser.add_argument("--http_host", default="0.0.0.0")
    parser.add_argument("--http_port", type=int, default=7070)
    parser.add_argument(
        "--profanity_file",
        default=None,
        help="Path to profanity words list. It should be a JSON file containing a list of words.",
    )
    args = parser.parse_args()

    if args.quant == "4bit":
        assert args.gptq_checkpoint is not None, "Please specify a GPTQ checkpoint."

    if args.tokenizer_path is None:
        args.tokenizer_path = args.pretrained
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)

    if args.profanity_file is not None:
        censored_words = load_json(args.profanity_file)
    else:
        censored_words = []
    prompt_processor = ChatPromptProcessor(censored_words=censored_words)

    if args.quant == "4bit":
        with low_resource_init():
            config = AutoConfig.from_pretrained(args.pretrained)
            model = AutoModelForCausalLM(config)
        model = llama_load_quant(model, args.gptq_checkpoint, 4, args.gptq_group_size)
        model.cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained,
            load_in_8bit=(args.quant == "8bit"),
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
        if args.quant != "8bit":
            model.half()  # seems to fix bugs for some users.
        model.eval()

    config = uvicorn.Config(app, host=args.http_host, port=args.http_port)
    server = uvicorn.Server(config=config)
    server.run()


"""
python server.py /home/lcyab/data/models/experiments5/checkpoint/experiment5-2023-10-20-21-53-51/modeling/ --tokenizer_path /mnt/vepfs/lcxyc/leaderboard_models/Colossal-LLaMA-2-7b-base/
"""
