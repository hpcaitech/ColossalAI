"""What the NPU actually reports about memory, so the training metric can stop under-reporting.

`torch.npu.max_memory_allocated()` counts only tensors the caching allocator handed out. It excludes the
allocator's own reserved-but-idle blocks, the device runtime's context, HCCL buffers, and anything another
process holds on the same card. This prints every counter next to the device-level figure so the gap is visible.
"""

import torch

torch.npu.set_device(0)
print("has mem_get_info:", hasattr(torch.npu, "mem_get_info"))
print("has memory_reserved:", hasattr(torch.npu, "memory_reserved"))
print("has max_memory_reserved:", hasattr(torch.npu, "max_memory_reserved"))

if hasattr(torch.npu, "mem_get_info"):
    info = torch.npu.mem_get_info()
    print("mem_get_info ->", type(info).__name__, len(info), [f"{value / 2**30:.2f} GiB" for value in info])

def report(label):
    allocated = torch.npu.memory_allocated() / 2**30
    reserved = torch.npu.memory_reserved() / 2**30 if hasattr(torch.npu, "memory_reserved") else float("nan")
    peak = torch.npu.max_memory_allocated() / 2**30
    peak_reserved = (
        torch.npu.max_memory_reserved() / 2**30 if hasattr(torch.npu, "max_memory_reserved") else float("nan")
    )
    free, total = torch.npu.mem_get_info()
    used = (total - free) / 2**30
    print(f"{label:22s} allocated {allocated:6.2f}  reserved {reserved:6.2f}  peak {peak:6.2f}  "
          f"peak_reserved {peak_reserved:6.2f}  device_used {used:6.2f} / {total / 2**30:.2f}")


report("after init")
x = torch.empty(4 * 1024**3 // 2, dtype=torch.float16, device="npu:0")  # 4 GiB
report("after 4 GiB tensor")
del x
report("after free")
