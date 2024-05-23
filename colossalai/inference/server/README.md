# Online Service
Colossal-Inference supports fast-api based online service. Simple completion and chat are both supported. Follow the commands below and you can simply construct a server with both completion and chat functionalities. For now we support `Llama2`,`Llama3` and `Baichuan2` model, etc. we will fullfill the blank quickly.

# API

## Completion
Completion api is used for single sequence request, like answer a question or complete words.
## Chat
Chat api is used for conversation-style request, which often includes dialogue participants(i.e. roles) and corresponding words. Considering the input data are very different from normal inputs, we introduce Chat-Template to match the data format in chat models.
### chat-template
Followed `transformers`, we add the chat-template argument. As chat models have been trained with very different formats for converting conversations into a single tokenizable string. Using a format that matches the training data is extremely important. This attribute(chat_template) is inclueded in HuggingFace tokenizers, containing a Jinja template that converts conversation histories into a correctly formatted string. You can refer to the [blog](https://huggingface.co/blog/chat-templates) for more information. We also provide a simple example temlate bellow. Both str or file style chat template are supported.
# Usage
## Args for customizing your server
The configuration for api server contains both serving interface and engine backend.
For Interface:
- `--host`: The host url on your device for the server.
- `--port`: The port for service
- `--model`: The model that backend engine uses, both path and transformers model card are supported.
- `--chat-template` The file path of chat template or the template string.
- `--response-role` The role that colossal-inference plays.
For Engine Backend:
- `--block_size`: The memory usage for each block.
- `--max_batch_size`: The max batch size for engine to infer. This changes the speed of inference,
- `--max_input_len`: The max input length of a request.
- `--max_output_len`: The output length of response.
- `--dtype` and `--use_cuda_kernel`: Deciding the precision and kernel usage.
For more detailed arguments, please refer to source code.

## Examples
```bash
# First, Lauch an API locally.
python3 -m colossalai.inference.server.api_server  --model path of your model --chat-template "{% for message in messages %}{{'<|im_start|>'+message['role']+'\n'+message['content']+'<|im_end|>'+'\n'}}{% endfor %}"

# Second, you can turn to the page `http://127.0.0.1:8000/docs` to check the api

# For completion service, you can invoke it
curl -X POST  http://127.0.0.1:8000/completion  -H 'Content-Type: application/json'  -d '{"prompt":"hello, who are you? "}'

# For chat service, you can invoke it
curl -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"messages":[{"role":"system","content":"you are a helpful assistant"},{"role":"user","content":"what is 1+1?"}]}'

# You can check the engine status now
curl http://localhost:8000/engine_check
```
