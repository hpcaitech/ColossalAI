#!/usr/bin/env bash

git reset --hard HEAD
mkdir -p ./all_dist
source activate base
conda create -n $1 -y python=$1
source activate $1
wget -q $2
ls | grep torch | xargs pip install
python setup.py bdist_wheel
mv ./dist/* ./all_dist
ls | grep torch | xargs rm
python setup.py clean
conda env remove -n $1


