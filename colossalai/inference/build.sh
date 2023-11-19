#!/usr/bin/env bash

# install triton 
pip install triton
pip install transformers

# install lightllm and flash-attention 
mkdir 3rdParty
cd 3rdParty
git clone https://github.com/ModelTC/lightllm 
cd lightllm
git checkout ece7b43f8a6dfa74027adc77c2c176cff28c76c8
pip install -e . 
cd ..

git clone -recursive https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install -e . 

cd ../../




