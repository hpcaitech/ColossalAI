prompt=$1
echo $prompt
python api_client.py \
       --host localhost \
       --port 8000 \
       --n 4 \
       --prompt "$prompt" \
       --stream
