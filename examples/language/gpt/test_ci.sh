set -x
pip install -r requirements.txt

cd gemini && bash test_ci.sh
# cd ../hybridparallelism && bash run.sh
