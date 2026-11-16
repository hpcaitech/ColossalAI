"""Keep each rank's process tree on the sockets its NPU is attached to.

The cluster has no CPU or memory pinning by default: every rank and every dataloader worker runs with
``Cpus_allowed_list: 0-191``, so decoded frames, augmentation temporaries and packed tensors are first-touched on
whichever socket happened to be free.  ``numa_maps`` of a single rank shows its 16 GiB spread over 2-5 domains.
Two machines in the six-node set (node5, node9) expose 4 NUMA domains instead of 8, and their ranks are exactly the
ones that arrive late at the end-of-backward collective, so the whole world waits for them.

The map is a semicolon-separated cpuset per *device index* (the order ``npu-smi`` lists them, which is also the order
``ASCEND_RT_VISIBLE_DEVICES=0,1,...,7`` selects).  ``train_sft.py`` applies entry ``LOCAL_RANK`` to itself at
startup; the dataloader workers are forked/spawned afterwards and inherit the mask, so their allocations stay local.

The cpuset is the whole *socket* the NPU sits on, not the 24-core NUMA domain, because two ranks share a socket:
pinning each rank to 24 cores would put ~30 threads there, while the socket's 48 cores hold 2 ranks comfortably.

Usage (from a node, with the container's environment)::

    COSMOS3_NUMA_PIN=$(python -m examples.worldmodel.cosmos3.numa_affinity --print-map)

and in the trainer, ``COSMOS3_NUMA_PIN`` set without ``--print-map``.  When the variable is unset nothing happens,
which keeps every existing recipe and benchmark bit-for-bit unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

PCI_PREFIX = "0000:"


def parse_cpulist(text: str) -> list[int]:
    """``"0-23,48,50-51"`` -> ``[0, ..., 23, 48, 50, 51]``."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def format_cpulist(cpus) -> str:
    """Inverse of :func:`parse_cpulist`: ``[0,1,2,5]`` -> ``"0-2,5"``."""
    ordered = sorted(set(int(c) for c in cpus))
    if not ordered:
        return ""
    spans: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        spans.append(f"{start}-{prev}" if prev != start else str(start))
        start = prev = cpu
    spans.append(f"{start}-{prev}" if prev != start else str(start))
    return ",".join(spans)


def parse_spec(spec: str) -> list[str]:
    """Split and validate a ``COSMOS3_NUMA_PIN`` value into one cpuset per device index."""
    entries = [entry.strip() for entry in spec.split(";")]
    if not entries or not any(entries):
        raise ValueError("empty NUMA pin map")
    for entry in entries:
        if not entry:
            raise ValueError(f"empty entry in NUMA pin map {spec!r}")
        parse_cpulist(entry)
    return entries


def affinity_for_rank(spec: str, local_rank: int) -> str:
    """The cpuset this local rank should run on."""
    entries = parse_spec(spec)
    return entries[local_rank % len(entries)]


def parse_device_buses(npu_smi_text: str) -> list[str]:
    """Ordered PCI bus ids from ``npu-smi info``, one per device index."""
    buses: list[str] = []
    for line in npu_smi_text.splitlines():
        fields = [field.strip() for field in line.split("|")]
        for field in fields:
            if field.startswith(PCI_PREFIX) and ":" in field[len(PCI_PREFIX) :]:
                buses.append(field)
                break
    return buses


def _read(path: Path) -> str:
    return path.read_text().strip()


def _socket_cpus(domain: int, sysfs: Path) -> list[int]:
    """Every CPU on the physical package that owns NUMA ``domain`` (one socket, all of its domains)."""
    domains = sorted(int(p.name[len("node") :]) for p in sysfs.glob("devices/system/node/node*"))
    packages: dict[int, int] = {}
    for other in domains:
        cpus = parse_cpulist(_read(sysfs / f"devices/system/node/node{other}/cpulist"))
        package = int(_read(sysfs / f"devices/system/cpu/cpu{min(cpus)}/topology/physical_package_id"))
        packages[other] = package
    wanted = packages[domain]
    cpus: list[int] = []
    for other, package in packages.items():
        if package == wanted:
            cpus.extend(parse_cpulist(_read(sysfs / f"devices/system/node/node{other}/cpulist")))
    return cpus


def build_map(device_buses: list[str], sysfs: Path | str = "/sys") -> list[str]:
    """One socket cpuset per device index, from each device's PCI NUMA node."""
    root = Path(sysfs)
    entries: list[str] = []
    for bus in device_buses:
        # npu-smi prints the bus id in upper case ("0000:C1:00.0"); sysfs names it in lower case.
        domain = int(_read(root / f"bus/pci/devices/{bus.lower()}/numa_node"))
        if domain < 0:
            raise ValueError(f"device {bus} reports no NUMA node")
        entries.append(format_cpulist(_socket_cpus(domain, root)))
    return entries


def apply_affinity(spec: str | None = None, local_rank: int | None = None) -> tuple[bool, str]:
    """Pin the current process to its rank's cpuset. No-op (returns False) when ``COSMOS3_NUMA_PIN`` is unset."""
    spec = spec if spec is not None else os.environ.get("COSMOS3_NUMA_PIN", "")
    if not spec:
        return False, "COSMOS3_NUMA_PIN unset, no pinning"
    if not hasattr(os, "sched_setaffinity"):  # macOS and other non-Linux hosts
        return False, "sched_setaffinity unavailable on this platform, no pinning"
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cpuset = affinity_for_rank(spec, local_rank)
    cpus = parse_cpulist(cpuset)
    os.sched_setaffinity(0, cpus)
    return True, f"local rank {local_rank} pinned to cpus {cpuset} ({len(cpus)} cores)"


def main(argv: list[str] | None = None) -> int:
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print-map", action="store_true", help="print the COSMOS3_NUMA_PIN value for this node")
    parser.add_argument("--npu-smi", default="npu-smi", help="npu-smi executable")
    args = parser.parse_args(argv)
    if not args.print_map:
        parser.error("nothing to do; pass --print-map")
    text = subprocess.run([args.npu_smi, "info"], capture_output=True, text=True, check=True).stdout
    buses = parse_device_buses(text)
    if not buses:
        raise SystemExit("npu-smi info listed no devices")
    print(";".join(build_map(buses)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
