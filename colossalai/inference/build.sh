#!/usr/bin/env bash

# install triton 
pip install triton
pip install transformers

# install lightllm and flash-attention 
mkdir 3rdParty
cd 3rdParty
git clone https://github.com/ModelTC/lightllm 
cd lightllm
git checkout 28c1267cfca536b7b4f28e921e03de735b003039
pip install -e . 
cd ..

git clone -recursive https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install -e . 

cd ../../




