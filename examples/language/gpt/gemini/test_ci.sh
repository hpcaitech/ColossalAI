set -x
$(cd `dirname $0`;pwd)
export TRAIN_STEP=4

for MODEL_TYPE in "gpt2_medium"; do
  for DISTPLAN in "CAI_Gemini"; do
    for BATCH_SIZE in 2; do
      for GPUNUM in 1 4; do
        for TPDEGREE in 1 2; do
          if [ ${TPDEGREE} -gt ${GPUNUM} ]; then
            continue
          fi
          for PLACEMENT in "cpu" "auto"; do
            MODEL_TYPE=${MODEL_TYPE} DISTPLAN=${DISTPLAN} BATCH_SIZE=${BATCH_SIZE} GPUNUM=${GPUNUM} TPDEGREE=${TPDEGREE} PLACEMENT=${PLACEMENT} \
            bash ./run_gemini.sh
          done
        done
      done
    done
  done

  for DISTPLAN in "zero1" "zero2"; do
    for BATCH_SIZE in 2; do
      for GPUNUM in 1 4; do
        for TPDEGREE in 1; do
          if [ ${TPDEGREE} -gt ${GPUNUM} ]; then
            continue
          fi
            MODEL_TYPE=${MODEL_TYPE} DISTPLAN=${DISTPLAN} BATCH_SIZE=${BATCH_SIZE} GPUNUM=${GPUNUM} TPDEGREE=${TPDEGREE}\
            bash ./run_gemini.sh
          done
        done
      done
    done
done
