pip install -r requirements.txt

# test colossalai
for TP in 1 2; do
    for PLACEMENT in "cpu" "cuda" "auto" "const"; do
        for SHARD in "True" "False"; do
            colossalai run --nproc_per_node=4 ./gemini/train_gpt_demo.py --steps 4 --distplan colossalai --tp_degree $TP --placement $PLACEMENT --shardinit $SHARD || exit 1
        done
    done
done

# test zero1&2
for DIST in "zero1" "zero2"; do
    colossalai run --nproc_per_node=4 ./gemini/train_gpt_demo.py --steps 4 --distplan $DIST || exit 1
done
