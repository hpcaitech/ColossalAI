:warning: **This content may be outdated since the major update of Colossal Chat. We will update this content soon.**

# ColossalAI on Ray

## Abstract

This is an experimental effort to run ColossalAI Chat training on Ray

## How to use?

### 1. Setup Ray clusters

Please follow the official [Ray cluster setup instructions](https://docs.ray.io/en/latest/cluster/getting-started.html) to setup an cluster with GPU support. Record the cluster's api server endpoint, it should be something similar to http://your.head.node.addrees:8265

### 2. Clone repo

Clone this project:

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
```

### 3. Submit the ray job

```shell
python applications/Chat/examples/community/ray/ray_job_script.py http://your.head.node.addrees:8265
```

### 4. View your job on the Ray Dashboard

Open your ray cluster dashboard http://your.head.node.addrees:8265 to view your submitted training job.
