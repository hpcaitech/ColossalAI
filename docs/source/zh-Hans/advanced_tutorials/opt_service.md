# Colossal-AI使用指南：5分钟搭建在线OPT服务

## 介绍

本指导手册将说明如何利用[Colossal-AI](https://github.com/hpcaitech/ColossalAI)搭建您自己的OPT服务。

## Colossal-AI 推理概述
Colossal-AI 提供了一个推理子系统 [Energon-AI](https://github.com/hpcaitech/EnergonAI)， 这是一个基于Colossal-AI的服务系统，拥有以下特性：

- **大模型并行：** 在Colossal-AI的张量并行和流水线并行策略的帮助下，Colossal-AI的推理可实现大模型的高效并行推理。
- **预构建大模型：** Colossal-AI提供热门模型的预构建部署，例如OPT。其支持用于生成任务和加载检查点的缓存技术。
- **引擎封装：** Colossal-AI中有一个抽象层被称作引擎。其将单实例多设备(SIMD) 执行与远程过程调用封装在一起。
- **在线服务系统：** 基于FastAPI，用户可以快速启动分布式推理的网络服务。 在线服务对生成任务进行了特殊优化。它采用left padding和bucket batching两种技术来提高效率。

## 基本用法

1. 下载OPT模型

想要快速发布分布式推理服务，您从[此处](https://huggingface.co/patrickvonplaten/opt_metaseq_125m/blob/main/model/restored.pt)下载OPT-125M。有关加载其他体量模型的详细方法，您可访问[此处](https://github.com/hpcaitech/EnergonAI/tree/main/examples/opt/script)。

2. 准备提前构建的服务镜像

从dockerhub拉取一个已经安装Colossal-AI推理的docker镜像。

```bash
docker pull hpcaitech/energon-ai:latest
```

3. 发布HTTP服务

若想发布服务，我们需要准备python脚本来描述模型的类型和相关的部署，以及HTTP服务的设置。 我们为您提供了一组[示例](https://github.com/hpcaitech/EnergonAI/tree/main/examples])。 我们将在本指导手册中使用[OPT 示例](https://github.com/hpcaitech/EnergonAI/tree/main/examples/opt)。
服务的入口是一个bash脚本 server.sh。
本服务的配置文件参考 opt_config.py，该文件定义了模型的类型、 检查点文件路径、并行策略和http设置。您能按照您的需求来修改这些设置。
例如，将模型的大小设置为opt_125M，将正确的检查点路径按照如下设置：

```bash
model_class = opt_125M
checkpoint = 'your_file_path'
```

将张量并行度设置为您的gpu数量。

```bash
tp_init_size = #gpu
```

现在，我们就能利用docker发布一个服务。您能在`/model_checkpoint` 和 `/config`路径下找到检查点文件和配置文件。


```bash
export CHECKPOINT_DIR="your_opt_checkpoint_path"
# the ${CONFIG_DIR} must contain a server.sh file as the entry of service
export CONFIG_DIR="config_file_path"

docker run --gpus all  --rm -it -p 8020:8020 -v ${CHECKPOINT_DIR}:/model_checkpoint -v ${CONFIG_DIR}:/config --ipc=host energonai:latest
```

接下来，您就可以在您的浏览器中打开 `https://[IP-ADDRESS]:8020/docs#` 进行测试。

## 高级特性用法

1. 批处理优化

若想使用我们的高级批处理技术来批量收集多个查询，您可以将executor_max_batch_size设置为最大批处理大小。 请注意，只有具有相同 top_k、top_p 和温度的解码任务才能一起批处理。

```
executor_max_batch_size = 16
```

所有的查询将进入FIFO队列。解码步数小于或等于队列头部解码步数的所有连续查询可以一起批处理。  应用左填充以确保正确性。 executor_max_batch_size 不应该过大，从而确保批处理不会增加延迟。 以opt-30b为例， `executor_max_batch_size=16` 合适，但对于opt-175b而言， `executor_max_batch_size=4` 更合适。

2. 缓存优化

对于每一个独立的服务过程，您能将最近的多个查询结果缓存在一起。在config.py中设置 cache_size 和 cache_list_size。缓存的大小应为缓存的查询数目。cache_list_size 应为每次查询存储的结果数。一个随机缓存的结果将会被返回。当缓存已满，LRU策略被用于清理缓存过的查询。cache_size=0意味着不缓存。

```
cache_size = 50
cache_list_size = 2
```
