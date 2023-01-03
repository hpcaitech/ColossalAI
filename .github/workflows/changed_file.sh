#!/bin/bash

res=$1
listed_res=($res)
echo "${listed_res} is current directory"
for sub_dir in ${listed_res[*]}; do
  pushd "examples/${sub_dir}" > /dev/null
      sh test_ci.sh
  popd < /dev/null
  echo "${sub_dir} has been executed"
done
