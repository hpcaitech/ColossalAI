# Overview

This is an example showing how to run OPT generation. The OPT model is implemented using ColossalAI.

It supports tensor parallelism, batching and caching.

## 🚀Quick Start
1. Run inference with OPT 125M
```bash
docker hpcaitech/tutorial:opt-inference
docker run -it --rm --gpus all --ipc host -p 7070:7070 hpcaitech/tutorial:opt-inference
```
2. Start the http server inside the docker container with tensor parallel size 2
```bash
python opt_fastapi.py opt-125m --tp 2 --checkpoint /data/opt-125m
```

# How to run

Run OPT-125M:
```shell
python opt_fastapi.py opt-125m
```

It will launch a HTTP server on `0.0.0.0:7070` by default and you can customize host and port. You can open `localhost:7070/docs` in your browser to see the openapi docs.

## Configure

### Configure model
```shell
python opt_fastapi.py <model>
```
Available models: opt-125m, opt-6.7b, opt-30b, opt-175b.

### Configure tensor parallelism
```shell
python opt_fastapi.py <model> --tp <TensorParallelismWorldSize>
```
The `<TensorParallelismWorldSize>` can be an integer in `[1, #GPUs]`. Default `1`.

### Configure checkpoint
```shell
python opt_fastapi.py <model> --checkpoint <CheckpointPath>
```
The `<CheckpointPath>` can be a file path or a directory path. If it's a directory path, all files under the directory will be loaded.

### Configure queue
```shell
python opt_fastapi.py <model> --queue_size <QueueSize>
```
The `<QueueSize>` can be an integer in `[0, MAXINT]`. If it's `0`, the request queue size is infinite. If it's a positive integer, when the request queue is full, incoming requests will be dropped (the HTTP status code of response will be 406).

### Configure batching
```shell
python opt_fastapi.py <model> --max_batch_size <MaxBatchSize>
```
The `<MaxBatchSize>` can be an integer in `[1, MAXINT]`. The engine will make batch whose size is less or equal to this value.

Note that the batch size is not always equal to `<MaxBatchSize>`, as some consecutive requests may not be batched.

### Configure caching
```shell
python opt_fastapi.py <model> --cache_size <CacheSize> --cache_list_size <CacheListSize>
```
This will cache `<CacheSize>` unique requests. And for each unique request, it cache `<CacheListSize>` different results. A random result will be returned if the cache is hit.

The `<CacheSize>` can be an integer in `[0, MAXINT]`. If it's `0`, cache won't be applied. The `<CacheListSize>` can be an integer in `[1, MAXINT]`.

### Other configurations
```shell
python opt_fastapi.py -h
```

# How to benchmark
```shell
cd benchmark
locust
```

Then open the web interface link which is on your console.

# Pre-process pre-trained weights

## OPT-66B
See [script/processing_ckpt_66b.py](./script/processing_ckpt_66b.py).

## OPT-175B
See [script/process-opt-175b](./script/process-opt-175b/).
