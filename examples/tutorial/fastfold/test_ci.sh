set -euxo pipefail

git clone https://github.com/hpcaitech/FastFold
cd FastFold
pip install -r requirements/requirements.txt
python setup.py install
pip install -r requirements/test_requirements.txt
cd ..

python inference.py
