#!/usr/bin/env bash

method=${1}
url=${2}
filename=${3}
cuda_version=${4}
python_version=${5}
torch_version=${6}
flags=${@:7}

echo $flags

exit

git reset --hard HEAD
mkdir -p ./all_dist
source activate base
conda create -n $python_version -y python=$python_version
source activate $1

if [ $1 == "pip" ]
then
    wget -nc -q -O ./pip_wheels/$filename $url
    pip install ./pip_wheels/$filename
    
elif [ $1 == 'conda' ]
then
    conda install pytorch==$torch_version cudatoolkit=$cuda_version $@
    echo You may go to the party but be back before midnight.
else
    echo Invalid installation method
    exit
fi

python setup.py bdist_wheel
mv ./dist/* ./all_dist
python setup.py clean
conda env remove -n $python_version


