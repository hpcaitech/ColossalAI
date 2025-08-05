#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Example script demonstrating the enhanced multi-node training setup for Kubernetes environments.

This script addresses the issue from GitHub issue #6349 where multi-node training
gets stuck during distributed initialization in Kubernetes environments.

Usage:
    # In Kubernetes with proper environment variables set:
    python k8s_multinode_example.py

    # Or with torchrun (enhanced command):
    torchrun --nnodes=4 --nproc_per_node=8 --node_rank=$NODE_RANK \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv_id=$JOB_ID k8s_multinode_example.py
"""

import os
import sys
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn

import colossalai
from colossalai.utils import diagnose_distributed_issues, setup_k8s_networking, validate_k8s_environment


def create_simple_model():
    """Create a simple model for testing distributed training."""
    return nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))


def test_distributed_operations():
    """Test basic distributed operations to verify setup."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] Testing distributed operations...")

    # Test all-reduce
    tensor = torch.ones(1).cuda() * rank
    original_value = tensor.item()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(world_size))

    if tensor.item() == expected_sum:
        print(f"[Rank {rank}] ✓ All-reduce test passed: {original_value} -> {tensor.item()} (expected: {expected_sum})")
        return True
    else:
        print(f"[Rank {rank}] ✗ All-reduce test failed: {original_value} -> {tensor.item()} (expected: {expected_sum})")
        return False


def run_training_simulation():
    """Simulate a simple training loop to verify distributed setup."""
    rank = dist.get_rank()
    dist.get_world_size()

    print(f"[Rank {rank}] Starting training simulation...")

    # Create model and move to GPU
    model = create_simple_model().cuda()
    model = nn.parallel.DistributedDataParallel(model)

    # Simple optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Simulate training steps
    for step in range(5):
        # Generate random batch
        batch_size = 32
        inputs = torch.randn(batch_size, 100).cuda()
        targets = torch.randint(0, 10, (batch_size,)).cuda()

        # Forward pass
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"[Step {step + 1}] Loss: {loss.item():.4f}")

        # Synchronize all processes
        dist.barrier()

    print(f"[Rank {rank}] Training simulation completed successfully!")
    return True


def main():
    """Main function demonstrating enhanced multi-node training setup."""
    print(f"=== ColossalAI Enhanced Multi-Node Training Example ===")
    print(f"Start time: {datetime.now()}")
    print(f"Process ID: {os.getpid()}")

    # Step 1: Validate Kubernetes environment
    print("\n1. Validating Kubernetes environment...")
    try:
        env_vars = validate_k8s_environment()
        print(f"✓ Environment validation passed! Found {len(env_vars)} variables.")

        # Print key environment variables for debugging
        print("Key environment variables:")
        for var in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK", "NODE_RANK"]:
            if var in env_vars:
                print(f"  {var}: {env_vars[var]}")
    except RuntimeError as e:
        print(f"✗ Environment validation failed: {e}")
        return 1

    # Step 2: Setup Kubernetes networking
    print("\n2. Setting up Kubernetes networking...")
    setup_k8s_networking()
    print("✓ Networking configuration applied")

    # Step 3: Run diagnostics if enabled
    if os.environ.get("DEBUG_DISTRIBUTED", "0") == "1":
        print("\n3. Running distributed training diagnostics...")
        diagnosis = diagnose_distributed_issues()

        print("Diagnosis results:")
        for check, status in diagnosis.items():
            if check == "recommendations":
                continue
            status_str = "PASS" if status else "FAIL"
            print(f"  {check}: {status_str}")

        if diagnosis["recommendations"]:
            print("Recommendations:")
            for i, rec in enumerate(diagnosis["recommendations"], 1):
                print(f"  {i}. {rec}")

    # Step 4: Initialize ColossalAI with enhanced error handling
    print("\n4. Initializing ColossalAI distributed training...")
    try:
        start_time = time.time()
        colossalai.launch_from_torch(verbose=True)
        init_time = time.time() - start_time

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        print(f"✓ ColossalAI initialization successful!")
        print(f"  Initialization time: {init_time:.2f} seconds")
        print(f"  Global rank: {rank}/{world_size}")
        print(f"  Local rank: {local_rank}")
        print(f"  Device: {torch.cuda.current_device()}")

    except Exception as e:
        print(f"✗ ColossalAI initialization failed: {e}")
        return 1

    # Step 5: Test distributed operations
    print("\n5. Testing distributed operations...")
    try:
        if test_distributed_operations():
            print("✓ Distributed operations test passed")
        else:
            print("✗ Distributed operations test failed")
            return 1
    except Exception as e:
        print(f"✗ Distributed operations test error: {e}")
        return 1

    # Step 6: Run a simple training simulation
    print("\n6. Running training simulation...")
    try:
        if run_training_simulation():
            print("✓ Training simulation completed successfully")
        else:
            print("✗ Training simulation failed")
            return 1
    except Exception as e:
        print(f"✗ Training simulation error: {e}")
        return 1

    # Success!
    print(f"\n=== Multi-Node Training Setup Successful! ===")
    print(f"End time: {datetime.now()}")
    print(f"Process {dist.get_rank()}/{dist.get_world_size()} completed successfully")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up distributed state
        if dist.is_initialized():
            dist.destroy_process_group()
