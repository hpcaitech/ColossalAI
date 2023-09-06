set -xe
pip uninstall colossalai
pip install -r requirements.txt

bash infer.sh
