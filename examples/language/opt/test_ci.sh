for GPUNUM in 2 1
do
env BS=2 MODEL="125m" GPUNUM=$GPUNUM bash ./run_gemini.sh
done
