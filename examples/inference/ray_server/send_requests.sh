prompt=$1
echo $prompt
python client.py \
       --host localhost \
       --port 8000 \
       --n 4 \
       --prompt "$prompt" \
       --stream
