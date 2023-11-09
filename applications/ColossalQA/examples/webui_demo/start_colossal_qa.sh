#!/bin/bash
cleanup() {
    echo "Caught Signal ... cleaning up."
    pkill -P $$  # kill all subprocess of this script
    exit 1       # exit script
}
# 'cleanup' is trigered when receive SIGINT(Ctrl+C) OR SIGTERM(kill) signal
trap cleanup INT TERM

# Disable your proxy
# unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# Path to store knowledge base(Home Directory by default)
export TMP=$HOME

# Use m3e as embedding model
export EMB_MODEL="m3e"  # moka-ai/m3e-base model will be download automatically
# export EMB_MODEL_PATH="PATH_TO_LOCAL_CHECKPOINT/m3e-base"  # you can also specify the local path to embedding model

# Choose a backend LLM
# - ChatGLM2
# export CHAT_LLM="chatglm2"  
# export CHAT_LLM_PATH="PATH_TO_LOCAL_CHECKPOINT/chatglm2-6b"

# - ChatGPT
export CHAT_LLM="chatgpt"
# Auth info for OpenAI API
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# - Pangu
# export CHAT_LLM="pangu" 
# # Auth info for Pangu API
# export URL=""
# export USERNAME=""
# export PASSWORD=""
# export DOMAIN_NAME=""

# Run server.py and colossalqa_webui.py in the background
python server.py &
python webui.py &

# Wait for all processes to finish
wait
