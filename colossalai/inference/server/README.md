# Online Service
Colossal-Inference supports fast-api based online service. Simple completion and chat are both supported. Follow the commands below and you can simply construct a server with both completion and chat functionalities. For now we support `Llama`,`Glide_Llama` and `baichuan` model, we will fullfill the blank quickly.

# Usage
```bash
# First, Lauch an API locally.
python3 -m colossalai.inference.server.api_server  --model path of your model --chat-template "{% for message in messages %}{{'<|im_start|>'+message['role']+'\n'+message['content']+'<|im_end|>'+'\n'}}{% endfor %}"

# Second, you can turn to the page `http://127.0.0.1:8000/docs` to check the api

# For completion service, you can invoke it
curl -X POST  http://127.0.0.1:8000/completion  -H 'Content-Type: application/json'  -d '{"prompt":"hello, who are you? ","stream":"False"}'

# For chat service, you can invoke it
curl -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"messages":[{"role":"system","content":"you are a helpful assistant"},{"role":"user","content":"what is 1+1?"}],"stream":"False"}'
```
We also support `streaming` Json output, simply change the `stream` to `True` in the request body. However, it is for now a simplification of streaming response, updates will come soon!
