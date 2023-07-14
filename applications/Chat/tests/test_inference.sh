set -xue

BASE_DIR=$(dirname $(dirname $(realpath $BASH_SOURCE)))
EXAMPLES_DIR=$BASE_DIR/examples

echo "[Test]: testing inference ..."

for model in 'gpt2' 'bloom' 'opt' 'llama'; do
    python $EXAMPLES_DIR/inference.py --model $model
done
