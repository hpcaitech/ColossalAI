#!/bin/bash

res=$1
res1=($res)
echo "${res1} is current location"
for ii in ${res1[*]}; do
  pushd "examples/${ii}" > /dev/null
      sh test_ci.sh
  popd < /dev/null
  echo "${ii} has been executed"
done
