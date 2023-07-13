set -xue

for model in 'gpt2' 'bloom' 'opt' 'llama'; do
    python inference.py --model $model
done
