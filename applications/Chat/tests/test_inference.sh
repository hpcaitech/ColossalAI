set -xue

BASE_DIR=$(dirname $(dirname $(realpath $BASH_SOURCE)))
EXAMPLES_DIR=$BASE_DIR/examples

echo "[Test]: testing inference ..."

# HACK: skip llama due to oom
for model in 'gpt2' 'bigscience/bloom-560m' 'facebook/opt-350m'; do
    python $EXAMPLES_DIR/inference.py --model_path $model --io dummy --max_new_tokens 20
done
