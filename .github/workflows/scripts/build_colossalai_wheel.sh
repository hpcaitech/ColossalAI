#!/usr/bin/env bash

method=${1}
url=${2}
filename=${3}
cuda_version=${4}
python_version=${5}
torch_version=${6}
flags=${@:7}

git reset --hard HEAD
mkdir -p ./all_dist
source activate base
conda create -n $python_version -y python=$python_version
source activate $python_version

if [ $1 == "pip" ]
then
    wget -nc -q -O ./pip_wheels/$filename $url
    pip install ./pip_wheels/$filename

elif [ $1 == 'conda' ]
then
    conda install pytorch==$torch_version cudatoolkit=$cuda_version $flags
else
    echo Invalid installation method
    exit
fi

if [ $cuda_version == "10.2" ]
then
    cp -r cub-1.8.0/cub/ colossalai/kernel/cuda_native/csrc/kernels/include/
fi

python setup.py bdist_wheel
mv ./dist/* ./all_dist
# must remove build to enable compilation for
# cuda extension in the next build
rm -rf ./build
python setup.py clean
conda deactivate
conda env remove -n $python_version
