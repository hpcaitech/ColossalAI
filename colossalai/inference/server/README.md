# Online Service
Colossal-Inference supports fast-api based online service. Simple completion and chat are both supported. Follow the commands below and you can simply construct a server with both completion and chat functionalities. For now we support `Llama`,`Glide_Llama` and `baichuan` model, we will fullfill the blank quickly.

# API

## Completion
Completion api is used for single sequence request, like answer a question or complete words.
## Chat
Chat api is used for conversation-style request, which often includes dialogue participants(i.e. roles) and corresponding words. Considering the input data are very different from normal inputs, we introduce Chat-Template to match the data format in chat models.
### chat-template
Followed `transformers`, we add the chat-template argument. As chat models have been trained with very different formats for converting conversations into a single tokenizable string. Using a format that matches the training data is extremely important. This attribute(chat_template) is inclueded in HuggingFace tokenizers, containing a Jinja template that converts conversation histories into a correctly formatted string. You can refer to the [blog](https://huggingface.co/blog/chat-templates) for more information. We also provide a simple example temlate bellow. Both str or file style chat template are supported.
# Usage
First, launch your own api server. You can assign the host and port by setting `--host` and `--port` argument. The
necessary argument is `--model`, which can be both path or the transformer string. If you want to utilize the chat api, you can add `--chat-template` and `--response-role`(the role that colossal-inference plays).
If you want to  customize the engine backend, consider change `block_size`(memory usage for each block), `max_batch_size`(changes the speed of inference), `max_input_len`(this will restrict the input length of request),
`max_output_len`(restricts the output length of response). You can also change `dtype` and `use_cuda_kernel` for deciding the precision and kernel usage.(For the detailed arguments, please refer to source code)

Second, you can turn to the page in `/docs` to check the api.

Last, start using both apis.
## Examples
```bash
# First, Lauch an API locally.
python3 -m colossalai.inference.server.api_server  --model path of your model --chat-template "{% for message in messages %}{{'<|im_start|>'+message['role']+'\n'+message['content']+'<|im_end|>'+'\n'}}{% endfor %}"

# Second, you can turn to the page `http://127.0.0.1:8000/docs` to check the api

# For completion service, you can invoke it
curl -X POST  http://127.0.0.1:8000/completion  -H 'Content-Type: application/json'  -d '{"prompt":"hello, who are you? "}'

# For chat service, you can invoke it
curl -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"messages":[{"role":"system","content":"you are a helpful assistant"},{"role":"user","content":"what is 1+1?"}]}'
```
