#!/bin/bash

#argument1: model_path

# launch server
model_path=${1:-"lmsys/vicuna-7b-v1.3"}
chat_template="{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
echo "Model Path: $model_path"
echo "Chat Tempelate" "${chat_template}"
echo "Starting server..."
python -m colossalai.inference.server.api_server --model $model_path --chat-template "${chat_template}" &
SERVER_PID=$!

# waiting time
sleep 60

# Run Locust
echo "Starting Locust..."
echo "The test will automatically begin, you can turn to http://0.0.0.0:8089 for more information."
echo "Test completion api first"
locust -f locustfile.py -t 300 --tags online-generation --host http://127.0.0.1:8000 --autostart --users 300 --stop-timeout 10
echo "Test chat api"
locust -f locustfile.py -t 300 --tags online-chat --host http://127.0.0.1:8000 --autostart --users 300 --stop-timeout 10
# kill Server
echo "Stopping server..."
kill $SERVER_PID

echo "Test and server shutdown completely"
