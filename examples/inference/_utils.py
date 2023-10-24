def print_perf_stats(latency_set, config, bs, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        avg = sum(latency_set) / count
        num_layers = (
            getattr(config, "num_layers") if hasattr(config, "num_layers") else getattr(config, "num_hidden_layers")
        )
        num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
        num_bytes = 2  # float16

        print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        print("Avg BW: {0:8.2f} GB/s".format(1 / avg * num_parameters * num_bytes / 1e9))
        print("Avg flops: {0:8.2f} TFlops/s".format(1 / avg * num_parameters * num_bytes * bs / 1e12))
        print("Avg Throughput: tokens/s: {}".format((1000 / (avg * 1000)) * bs))
