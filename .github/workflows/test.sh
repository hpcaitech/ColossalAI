res=$1
res1=($res)
for ii in ${res1[*]}; do
  cd "examples/${ii}"
  sh test_ci.sh
  cd ../../..
  echo "${ii} haha"
done