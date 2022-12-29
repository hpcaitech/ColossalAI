for MODEL_NAME in "GPT2small"
do
for BATCH_SIZE in 8
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
echo "* benchmrking MODEL_NAME ${MODEL_NAME} BS ${BATCH_SIZE} BS ${BS} GPUNUM ${GPUNUM} TPDEGREE ${TPDEGREE}"
bash ./run.sh
echo "****************** Finished ***************************"
echo ""
echo ""
done
done
done
done
