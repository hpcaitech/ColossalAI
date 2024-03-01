"""
Doc:
    Feature:
    - FastAPI based http server for Colossal-Inference
    - Completion Service Supported
    Usage: (for local user)
    - First, Lauch an API locally. `python3 -m colossalai.inference.server.api_server  --model path of your llama2 model`
    - Second, you can turn to the page `http://127.0.0.1:8000/docs` to check the api
    - For completion service, you can invoke it by using `curl -X POST  http://127.0.0.1:8000/v1/completion  \
         -H 'Content-Type: application/json' \
         -d '{"prompt":"hello, who are you? ","stream":"False"}'`
"""


import argparse
import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

from colossalai.inference.config import InferenceConfig
from colossalai.inference.server.completion_service import CompletionServing
from colossalai.inference.server.utils import id_generator

from colossalai.inference.core.async_engine import AsyncInferenceEngine, InferenceEngine  # noqa

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None
supported_models_dict = {"Llama_Models": ("llama2-7b",)}
prompt_template_choices = ["llama", "vicuna"]


@app.get("/v0/models")
def get_available_models() -> Response:
    return JSONResponse(supported_models_dict)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    A request should be a JSON object with the following fields:
    - prompts: the prompts to use for the generation.
    - stream: whether to stream the results or not.
    - other fields:
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", None)

    request_id = id_generator()
    generation_config = get_generation_config(request_dict)
    results = engine.generate(request_id, prompt, generation_config=generation_config)

    # Streaming case
    def stream_results():
        for request_output in results:
            ret = {"text": request_output}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    for request_output in results:
        if request.is_disconnected():
            # Abort the request if the client disconnects.
            engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    ret = {"text": final_output}
    return JSONResponse(ret)


@app.post("/v1/completion")
async def create_completion(request: Request):
    request_dict = await request.json()
    generation_config = get_generation_config(request_dict)
    generator = await completion_serving.create_completion(request, generation_config)
    output = tokenizer.decode(generator.output_token_id)
    ret = {"request_id": generator.request_id, "text": output}
    return ret


def get_generation_config(request):
    generation_config = async_engine.engine.generation_config
    for arg in request:
        if hasattr(generation_config, arg):
            generation_config[arg] = request[arg]
    return generation_config


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
    parser.add_argument(
        "--prompt_template",
        choices=prompt_template_choices,
        default=None,
        help=f"Allowed choices are {','.join(prompt_template_choices)}. Default to None.",
    )

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

    inference_config = InferenceConfig.from_dict(vars(args))
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    async_engine = AsyncInferenceEngine(
        start_engine_loop=True, model=model, tokenizer=tokenizer, inference_config=inference_config
    )
    engine = async_engine.engine
    completion_serving = CompletionServing(async_engine, served_model=model.__class__.__name__)

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
