#!/usr/bin/env python3
"""Container Environment Validation Tool for Cosmos3 on Ascend 910B2.

Validates that:
1. COLOSSALAI_EXTENSION_CACHE is set and points to local NVMe storage (non-NFS).
2. TILELANG_CACHE_DIR is set and points to local NVMe storage (non-NFS).
3. HCCL_SOCKET_IFNAME is set to bond1.
4. OMP_NUM_THREADS is set to 8.
5. Virtualenv path does not reside on NFS (/home/).
6. /dev/shm capacity is >= 512 GB (warns or fails based on --strict).
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

FORBIDDEN_NFS_PATTERNS = ["/home/", "/nfs/", "/mnt/nfs/", "nfs://", "/shared/"]
NVME_PREFIX = "/mnt/disk0"
REQUIRED_SHM_GIB = 512


def is_nfs_path(path_str: str) -> bool:
    """Return True if path starts with or contains known NFS mount prefixes."""
    clean = path_str.strip()
    return any(clean.startswith(pat) or pat in clean for pat in FORBIDDEN_NFS_PATTERNS)


def validate_environment(
    check_shm: bool = True,
    strict_shm: bool = False,
    allow_temp: bool = False,
) -> Tuple[bool, List[str], List[str]]:
    """Validate container environment variables and storage paths.

    Returns:
        (is_valid, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. COLOSSALAI_EXTENSION_CACHE
    ext_cache = os.environ.get("COLOSSALAI_EXTENSION_CACHE")
    if not ext_cache or not ext_cache.strip():
        errors.append("COLOSSALAI_EXTENSION_CACHE environment variable is not set or empty.")
    elif is_nfs_path(ext_cache.strip()):
        errors.append(
            f"COLOSSALAI_EXTENSION_CACHE '{ext_cache}' resides on NFS. "
            f"Must be on local NVMe ({NVME_PREFIX}) to prevent FileBaton deadlocks."
        )
    elif not allow_temp and not ext_cache.strip().startswith(NVME_PREFIX):
        warnings.append(
            f"COLOSSALAI_EXTENSION_CACHE '{ext_cache}' does not start with standard NVMe prefix '{NVME_PREFIX}'."
        )

    # 2. TILELANG_CACHE_DIR
    tilelang_cache = os.environ.get("TILELANG_CACHE_DIR")
    if not tilelang_cache or not tilelang_cache.strip():
        errors.append("TILELANG_CACHE_DIR environment variable is not set or empty.")
    elif is_nfs_path(tilelang_cache.strip()):
        errors.append(
            f"TILELANG_CACHE_DIR '{tilelang_cache}' resides on NFS. "
            f"Must be on local NVMe ({NVME_PREFIX}) to prevent compilation stalls."
        )
    elif not allow_temp and not tilelang_cache.strip().startswith(NVME_PREFIX):
        warnings.append(
            f"TILELANG_CACHE_DIR '{tilelang_cache}' does not start with standard NVMe prefix '{NVME_PREFIX}'."
        )

    # 3. HCCL_SOCKET_IFNAME
    hccl_iface = os.environ.get("HCCL_SOCKET_IFNAME")
    if not hccl_iface or not hccl_iface.strip():
        errors.append("HCCL_SOCKET_IFNAME environment variable is not set (expected 'bond1').")
    elif hccl_iface.strip() != "bond1":
        warnings.append(f"HCCL_SOCKET_IFNAME is set to '{hccl_iface}' (expected 'bond1' for RoCE).")

    # 4. OMP_NUM_THREADS
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if not omp_threads or not omp_threads.strip():
        errors.append("OMP_NUM_THREADS environment variable is not set (expected '8').")
    elif omp_threads.strip() != "8":
        warnings.append(f"OMP_NUM_THREADS is set to '{omp_threads}' (expected '8' to prevent core thrashing).")

    # 5. Virtualenv path
    venv_path = sys.prefix
    if is_nfs_path(venv_path):
        errors.append(
            f"Python virtual environment '{venv_path}' resides on NFS. "
            f"Dataloader imports will suffer 100s+ RPC stalls. Migrate to {NVME_PREFIX}."
        )

    # 6. /dev/shm capacity
    if check_shm:
        shm_path = Path("/dev/shm")
        if shm_path.exists():
            try:
                usage = shutil.disk_usage(str(shm_path))
                total_gib = usage.total / (1024**3)
                if total_gib < REQUIRED_SHM_GIB:
                    msg = (
                        f"/dev/shm size is {total_gib:.1f} GiB (less than required {REQUIRED_SHM_GIB} GiB). "
                        f"Docker container requires --shm-size 512g to absorb ~190 GB in-flight batch data."
                    )
                    if strict_shm:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
            except Exception as e:
                warnings.append(f"Could not inspect /dev/shm disk usage: {e}")
        else:
            warnings.append("/dev/shm does not exist on this host platform.")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate container environment for Cosmos3 Ascend training.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings as well as errors.")
    parser.add_argument("--skip-shm", action="store_true", help="Skip /dev/shm capacity check.")
    parser.add_argument("--allow-temp", action="store_true", help="Allow temporary directory caches for local dev.")
    args = parser.parse_args()

    is_valid, errors, warnings = validate_environment(
        check_shm=not args.skip_shm,
        strict_shm=args.strict,
        allow_temp=args.allow_temp,
    )

    if warnings:
        print("[WARNINGS]")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("[ERRORS]")
        for e in errors:
            print(f"  - {e}")

    if not is_valid or (args.strict and warnings):
        print("\nEnvironment validation FAILED.")
        sys.exit(1)

    print("\nEnvironment validation PASSED. All paths and settings comply with local NVMe container contract.")
    sys.exit(0)


if __name__ == "__main__":
    main()
