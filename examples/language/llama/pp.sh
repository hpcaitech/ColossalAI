FLAGS=()

# Loop through all the arguments
for arg in "$@"
do
    # Check if the argument is in --key=value format
    if [[ $arg == --* ]]; then
        # Append the argument to FLAGS array, preserving the quoting
        FLAGS+=("$arg")
    fi
done

# Check if FLAGS array is not empty
if [ ${#FLAGS[@]} -ne 0 ]; then
    echo "Addings flags ${FLAGS[@]} from the command line."
fi

export OMP_NUM_THREADS=8
colossalai run --nproc_per_node 4  --master_port 29501 benchmark.py -p 3d --pp 4 -b 16 -g -x --pp_style interleaved --n_chunks 2 "${FLAGS[@]}"
