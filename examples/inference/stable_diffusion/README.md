## File Structure
```
|- sd3_generation.py: an example of how to use Colossalai Inference Engine to generate result by loading Diffusion Model.
|- compute_metric.py: compare the quality of images w/o some acceleration method like Distrifusion
|- benchmark_sd3.py: benchmark the performance of our InferenceEngine
|- run_benchmark.sh: run benchmark command
```
Note: compute_metric.py need some dependencies which need `pip install -r requirements.txt`, `requirements.txt` is in `examples/inference/stable_diffusion/`

## Run Inference

The provided example `sd3_generation.py` is an example to configure, initialize the engine, and run inference on provided model. We've added `DiffusionPipeline` as model class, and the script is good to run inference with StableDiffusion 3.

For a basic setting, you could run the example by:
```bash
colossalai run --nproc_per_node 1 sd3_generation.py -m PATH_MODEL -p "hello world"
```

Run multi-GPU inference (Patched Parallelism), as in the following example using 2 GPUs:
```bash
colossalai run --nproc_per_node 2 sd3_generation.py -m PATH_MODEL
```
