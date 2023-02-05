avai=true
for i in $(seq 0 7);
do
  gpu_used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$gpu_used" -le "10000" ] && avai=false
done
