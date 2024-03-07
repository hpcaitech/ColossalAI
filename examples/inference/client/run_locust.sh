#!/bin/bash

# launch server
model_path="/home/caidi/llama_model/"
echo "Starting server..."
python -m colossalai.inference.server.api_server --model $model_path &
SERVER_PID=$!

# 等待服务器启动
sleep 60

# 启动Locust
echo "Starting Locust..."
locust -f locustfile.py -t 300 --tags online-generation --host http://127.0.0.1:8000 --autostart --users 100

# 测试完成，终止server.py
kill $SERVER_PID
