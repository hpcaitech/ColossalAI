#!/bin/bash

#argument1: model_path

# launch server
model_path=${1:-"lmsys/vicuna-7b-v1.3"}
echo "Model Path: $model_path"
echo "Starting server..."
python -m colossalai.inference.server.api_server --model $model_path &
SERVER_PID=$!

# waiting time
sleep 60

# Run Locust
echo "Starting Locust..."
echo "The test will automatically begin, you can turn to http://0.0.0.0:8089 for more information."
locust -f locustfile.py -t 300 --tags online-generation --host http://127.0.0.1:8000 --autostart --users 100 --stop-timeout 10

# kill Server
echo "Stopping server..."
kill $SERVER_PID

echo "Test and server shutdown completely"
