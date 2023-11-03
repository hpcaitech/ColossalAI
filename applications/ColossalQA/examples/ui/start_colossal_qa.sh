#!/bin/bash
source set_env.sh  # Need to activate your virtual envirnment for colossalqa and set auth key for LLM API
cleanup() {
    echo "Caught Signal ... cleaning up."
    pkill -P $$  # kill all subprocess of this script
    exit 1       # exit script
}

# 'cleanup' is trigered when receive SIGINT(Ctrl+C) OR SIGTERM(kill) signal
trap cleanup INT TERM

# Run server.py and colossalqa_webui.py in the background
python server.py &
python colossalqa_webui.py &

# Wait for all processes to finish
wait
