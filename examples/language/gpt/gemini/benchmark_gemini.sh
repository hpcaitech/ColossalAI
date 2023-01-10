for MODEL_TYPE in "gpt2_medium"; do
  for DISPAN in "colossalai"; do
    for BATCH_SIZE in 16; do
      for GPUNUM in 1 2 4 8; do
        for TPDEGREE in 1 2 4 8; do
          if [ ${TPDEGREE} -gt ${GPUNUM} ]; then
            continue
          fi
          for PLACEMENT in "cpu" "auto"; do
            echo "****************** Begin ***************************"
            echo "+ benchmrking MODEL ${MODEL_TYPE} DISPAN ${DISPAN} GPU ${GPUNUM} BS ${BATCH_SIZE} TP ${TPDEGREE} POLICY ${PLACEMENT}"
            MODEL_TYPE=${MODEL_TYPE} DISPAN=${DISPAN} BATCH_SIZE=${BATCH_SIZE} GPUNUM=${GPUNUM} TPDEGREE=${TPDEGREE} PLACEMENT=${PLACEMENT} \
            bash ./run_gemini.sh
            echo "****************** Finished ***************************"
            echo ""
            echo ""
          done
        done
      done
    done
  done
done
