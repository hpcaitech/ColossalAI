import argparse
import json
from typing import Generator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

import colossalai
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.inference.server.utils import id_generator

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None
available_models = ["llama2-7b"]


@app.get("v0/models")
def available_models() -> Response:
    return JSONResponse(available_models)


@app.post("/generate")
def generate(request: Request) -> Response:
    """Generate completion for the request.

    A request should be a JSON object with the following fields:
    - prompts: the prompts to use for the generation.
    - stream: whether to stream the results or not.
    - other fields:
    """
    request_dict = request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    request_id = id_generator()

    results_generator = engine.generate(
        request_id,
        prompt,
    )

    # Streaming case
    def stream_results() -> Generator[bytes, None]:
        for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [prompt + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    for request_output in results_generator:
        if request.is_disconnected():
            # Abort the request if the client disconnects.
            engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


def add_engine_config(parser):
    parser.add_argument("--model", type=str, default="llama2-7b", help="name or path of the huggingface model to use")

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="model context length. If unspecified, " "will be automatically derived from the model.",
    )
    # Parallel arguments
    parser.add_argument(
        "--worker-use-ray",
        action="store_true",
        help="use Ray for distributed serving, will be " "automatically set when using more than 1 GPU",
    )

    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1, help="number of pipeline stages")

    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="number of tensor parallel replicas")

    # KV cache arguments
    parser.add_argument("--block-size", type=int, default=16, choices=[8, 16, 32], help="token block size")

    parser.add_argument("--max_batch_size", type=int, default=8, help="maximum number of batch size")

    # generation arguments
    parser.add_argument("--use_prompt_template", action="store_true", help="whether to use prompt template")

    # Quantization settings.
    parser.add_argument(
        "--quantization",
        "-q",
        type=str,
        choices=["awq", "gptq", "squeezellm", None],
        default=None,
        help="Method used to quantize the weights. If "
        "None, we first check the `quantization_config` "
        "attribute in the model config file. If that is "
        "None, we assume the model weights are not "
        "quantized and use `dtype` to determine the data "
        "type of the weights.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Always use eager-mode PyTorch. If False, "
        "will use eager mode and CUDA graph in hybrid "
        "for maximal performance and flexibility.",
    )
    return parser


def parse_args():
    parser = argparse.ArgumentParser(description="Colossal-Inference API server.")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path", type=str, default=None, help="FastAPI root_path when app is behind a path based routing proxy"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not "
        "specified, the model name will be the same as "
        "the huggingface name.",
    )
    parser = add_engine_config(parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    colossalai.launch(config={}, rank=0, world_size=1, host="localhost", port=args.port, backend="nccl")

    inference_config = InferenceConfig.from_cli_args(args)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = InferenceEngine(model=model, tokenizer=tokenizer, inference_config=inference_config)

    app.root_path = args.root_path
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
