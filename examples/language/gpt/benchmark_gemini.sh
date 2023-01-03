for MODEL_TYPE in "gpt2_medium"
do
for BATCH_SIZE in 16
do
for GPUNUM in 1 2 4 8
do
for TPDEGREE in 1 2 4 8
do
if [ ${TPDEGREE} -gt ${GPUNUM} ]
then
    continue
fi
echo "****************** Begin ***************************"
echo "* benchmrking MODEL_TYPE ${MODEL_TYPE} BS ${BATCH_SIZE} BS ${BS} GPUNUM ${GPUNUM} TPDEGREE ${TPDEGREE}"
MODEL_TYPE=${MODEL_TYPE} BATCH_SIZE=${BATCH_SIZE} GPUNUM=${GPUNUM} TPDEGREE=${TPDEGREE} bash ./run_gemini.sh
echo "****************** Finished ***************************"
echo ""
echo ""
done
done
done
done
