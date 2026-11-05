"""Check actual computation on every container-visible CUDA device."""


def probe_devices(torch, expected):
    if expected < 1:
        raise RuntimeError("The CUDA probe requires at least one allocated GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the container")
    count = torch.cuda.device_count()
    if count != expected:
        raise RuntimeError(f"Expected {expected} CUDA device(s), found {count}")
    names = []
    for index in range(expected):
        try:
            # Enumeration can succeed before device access is lost. Allocate,
            # launch a kernel and synchronize so that failures reach this gate.
            value = torch.ones(4, dtype=torch.float32, device=f"cuda:{index}").sum().item()
            torch.cuda.synchronize(index)
            if value != 4:
                raise RuntimeError(f"Unexpected CUDA probe result: {value}")
            names.append(torch.cuda.get_device_name(index))
        except Exception as error:
            raise RuntimeError(f"CUDA compute probe failed on cuda:{index}: {error}") from error
    return names
