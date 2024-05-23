# Online Service
Colossal-Inference supports fast-api based online service. Simple completion and chat are both supported. Follow the commands below and
you can simply construct a server with both completion and chat functionalities. For now we only support `Llama` model, we will fullfill
the blank quickly.

# Usage
```bash
# First, Lauch an API locally.
python3 -m colossalai.inference.server.api_server  --model path of your llama2 model --chat_template "{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"


# Second, you can turn to the page `http://127.0.0.1:8000/docs` to check the api

# For completion service, you can invoke it
curl -X POST  http://127.0.0.1:8000/completion  -H 'Content-Type: application/json'  -d '{"prompt":"hello, who are you? ","stream":"False"}'

# For chat service, you can invoke it
curl -X POST  http://127.0.0.1:8000/completion  -H 'Content-Type: application/json'  -d  '{"converation":
                [{"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "what is 1+1?"},],
                "stream": "False",}'
# If you just want to test a simple generation, turn to generate api
curl -X POST  http://127.0.0.1:8000/generate  -H 'Content-Type: application/json'  -d '{"prompt":"hello, who are you? ","stream":"False"}'

```
We also support streaming output, simply change the `stream` to `True` in the request body.
