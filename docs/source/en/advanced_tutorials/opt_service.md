# Build an online OPT service using Colossal-AI in 5 minutes

## Introduction

This tutorial shows how to build your own service with OPT with the help of [Colossal-AI](https://github.com/hpcaitech/ColossalAI).

## Colossal-AI Inference Overview
Colossal-AI provides an inference subsystem [Energon-AI](https://github.com/hpcaitech/EnergonAI), a serving system built upon Colossal-AI, which has the following characteristics:

- **Parallelism for Large-scale Models:** With the help of tensor parallel operations, pipeline parallel strategies from Colossal-AI, Colossal-AI inference enables efficient parallel inference for large-scale models.
- **Pre-built large models:** There are pre-built implementations for popular models, such as OPT. It supports a caching technique for the generation task and checkpoints loading.
- **Engine encapsulation：** There has an abstraction layer called an engine. It encapsulates the single instance multiple devices (SIMD) execution with the remote procedure call, making it act as the single instance single device (SISD) execution.
- **An online service system:** Based on FastAPI, users can launch a web service of a distributed inference quickly. The online service makes special optimizations for the generation task. It adopts both left padding and bucket batching techniques to improve efficiency.

## Basic Usage:

1. Download OPT model

To launch the distributed inference service quickly, you can download the OPT-125M from [here](https://huggingface.co/patrickvonplaten/opt_metaseq_125m/blob/main/model/restored.pt). You can get details for loading other sizes of models [here](https://github.com/hpcaitech/EnergonAI/tree/main/examples/opt/script).

2. Prepare a prebuilt service image

Pull a docker image from docker hub installed with Colossal-AI inference.

```bash
docker pull hpcaitech/energon-ai:latest
```

3. Launch an HTTP service

To launch a service, we need to provide python scripts to describe the model type and related configurations, and settings for the HTTP service.
We have provided a set of [examples](https://github.com/hpcaitech/EnergonAI/tree/main/examples]). We will use the [OPT example](https://github.com/hpcaitech/EnergonAI/tree/main/examples/opt) in this tutorial.
The entrance of the service is a bash script server.sh.
The config of the service is at opt_config.py, which defines the model type, the checkpoint file path, the parallel strategy, and http settings. You can adapt it for your own case.
For example, set the model class as opt_125M and set the correct checkpoint path as follows.

```bash
model_class = opt_125M
checkpoint = 'your_file_path'
```

Set the tensor parallelism degree the same as your gpu number.

```bash
tp_init_size = #gpu
```

Now, we can launch a service using docker. You can map the path of the checkpoint and directory containing configs to local disk path `/model_checkpoint` and `/config`.


```bash
export CHECKPOINT_DIR="your_opt_checkpoint_path"
# the ${CONFIG_DIR} must contain a server.sh file as the entry of service
export CONFIG_DIR="config_file_path"

docker run --gpus all  --rm -it -p 8020:8020 -v ${CHECKPOINT_DIR}:/model_checkpoint -v ${CONFIG_DIR}:/config --ipc=host energonai:latest
```

Then open `https://[IP-ADDRESS]:8020/docs#` in your browser to try out!


## Advance Features Usage:

1. Batching Optimization

To use our advanced batching technique to collect multiple queries in batches to serve, you can set the executor_max_batch_size as the max batch size. Note, that only the decoder task with the same top_k, top_p and temperature can be batched together.

```
executor_max_batch_size = 16
```

All queries are submitted to a FIFO queue. All consecutive queries whose number of decoding steps is less than or equal to that of the head of the queue can be batched together. Left padding is applied to ensure correctness. executor_max_batch_size should not be too large. This ensures batching won't increase latency. For opt-30b, `executor_max_batch_size=16` may be a good choice, while for opt-175b, `executor_max_batch_size=4` may be better.

2. Cache Optimization.

You can cache several recently served query results for each independent serving process. Set the cache_size and cache_list_size in config.py. The cache size is the number of queries cached. The cache_list_size is the number of results stored for each query. And a random cached result will be returned. When the cache is full, LRU is applied to evict cached queries. cache_size=0means no cache is applied.

```
cache_size = 50
cache_list_size = 2
```
