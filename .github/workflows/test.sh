#!/bin/bash

res=$1
res1=($res)
echo "${res1} is res1"
for ii in ${res1[*]}; do
  cd "examples/${ii}"
  sh test_ci.sh
  cd ../../..
  echo "${ii} haha"
done