# Kubernetes Multi-Node Training Guide for ColossalAI

This guide provides comprehensive instructions for setting up and troubleshooting multi-node distributed training with ColossalAI in Kubernetes environments.

## Problem Addressed

This solution addresses the common issue where multi-node training gets stuck during process group initialization in Kubernetes environments, particularly when using `torchrun` with the basic configuration.

## Key Improvements

### 1. Enhanced Initialization Logic
- **Connection readiness checks**: Non-master nodes wait for master to be ready
- **Configurable timeouts**: Extended timeouts for K8s networking delays
- **Better error messages**: Detailed diagnostics when initialization fails
- **Automatic retry mechanisms**: Robust handling of transient network issues

### 2. Kubernetes-Aware Configuration
- **DNS-based service discovery**: Use headless services for stable endpoints
- **Environment variable validation**: Comprehensive checks with helpful error messages
- **Network interface configuration**: Automatic setup for K8s networking
- **Debug logging**: Extensive logging for troubleshooting

## Quick Start

### 1. Enhanced Torchrun Command

Replace your basic torchrun command with this enhanced version:

```bash
# Old problematic command:
# torchrun --nnodes 4 --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node-rank $NODE_RANK scripts/diffusion/train.py configs/diffusion/train/demo.py --dataset.data-path modified_data.csv

# New enhanced command:
torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank=$NODE_RANK \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=$JOB_ID \
  --rdzv_conf="timeout=1800,read_timeout=120" \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/diffusion/train.py configs/diffusion/train/demo.py \
  --dataset.data-path modified_data.csv
```

### 2. Required Environment Variables

Set these environment variables in your Kubernetes deployment:

```yaml
env:
  # Essential variables
  - name: MASTER_ADDR
    value: "training-master-service.default.svc.cluster.local"
  - name: MASTER_PORT
    value: "29500"
  - name: WORLD_SIZE
    value: "32"  # 4 nodes * 8 GPUs
  - name: NODE_RANK
    valueFrom:
      fieldRef:
        fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
  - name: RDZV_ID
    value: "opensora-training-job-123"

  # Network configuration
  - name: NCCL_SOCKET_IFNAME
    value: "eth0"
  - name: GLOO_SOCKET_IFNAME
    value: "eth0"
  - name: NCCL_IB_DISABLE
    value: "1"
  - name: NCCL_DEBUG
    value: "WARN"  # Use "INFO" for debugging

  # ColossalAI timeout configuration
  - name: COLOSSALAI_DIST_TIMEOUT
    value: "1800"  # 30 minutes for initialization
  - name: COLOSSALAI_MASTER_READY_TIMEOUT
    value: "600"   # 10 minutes for master readiness
```

## Complete Kubernetes Setup

### Step 1: Create Headless Service

```bash
# Save as headless-service.yaml
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: training-master-service
  namespace: default
spec:
  clusterIP: None  # Headless service
  selector:
    app: colossalai-training
    role: master
  ports:
  - name: distributed-comm
    port: 29500
    targetPort: 29500
EOF
```

### Step 2: Create Training Job

```bash
# Save as training-job.yaml
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: opensora-multinode-training
  namespace: default
spec:
  parallelism: 4
  completions: 4
  completionMode: Indexed
  template:
    metadata:
      labels:
        app: colossalai-training
    spec:
      restartPolicy: Never
      containers:
      - name: training
        image: your-opensora-image:latest
        command:
        - /bin/bash
        - -c
        - |
          # Wait for pod startup synchronization
          sleep \$((RANDOM % 30 + 30))

          # Set additional environment variables
          export RANK=\$((JOB_COMPLETION_INDEX * 8))
          export LOCAL_RANK=0

          # Enhanced torchrun command
          torchrun \\
            --nnodes=4 \\
            --nproc_per_node=8 \\
            --node_rank=\$JOB_COMPLETION_INDEX \\
            --rdzv_backend=c10d \\
            --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \\
            --rdzv_id=opensora-training-\$JOB_NAME \\
            --rdzv_conf=timeout=1800,read_timeout=120 \\
            --master_addr=\$MASTER_ADDR \\
            --master_port=\$MASTER_PORT \\
            scripts/diffusion/train.py configs/diffusion/train/demo.py \\
            --dataset.data-path modified_data.csv
        env:
        - name: MASTER_ADDR
          value: "training-master-service.default.svc.cluster.local"
        - name: MASTER_PORT
          value: "29500"
        - name: WORLD_SIZE
          value: "32"
        - name: NODE_RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        - name: JOB_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['job-name']
        - name: NCCL_SOCKET_IFNAME
          value: "eth0"
        - name: GLOO_SOCKET_IFNAME
          value: "eth0"
        - name: NCCL_IB_DISABLE
          value: "1"
        - name: NCCL_DEBUG
          value: "WARN"
        - name: COLOSSALAI_DIST_TIMEOUT
          value: "1800"
        - name: COLOSSALAI_MASTER_READY_TIMEOUT
          value: "600"
        resources:
          requests:
            nvidia.com/gpu: 8
          limits:
            nvidia.com/gpu: 8
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 32Gi
EOF
```

## Using the Utility Functions

### Python Code Integration

```python
import colossalai
from colossalai.utils.k8s_distributed import (
    validate_k8s_environment,
    setup_k8s_networking,
    diagnose_distributed_issues
)

def main():
    # Validate K8s environment before starting
    try:
        env_vars = validate_k8s_environment()
        print(f"Environment validation passed: {len(env_vars)} variables found")
    except RuntimeError as e:
        print(f"Environment validation failed: {e}")
        return 1

    # Setup K8s networking
    setup_k8s_networking()

    # Run diagnostics if needed
    if os.environ.get("DEBUG_DISTRIBUTED", "0") == "1":
        diagnosis = diagnose_distributed_issues()
        print(f"Diagnosis results: {diagnosis}")

    # Initialize ColossalAI with enhanced error handling
    try:
        colossalai.launch_from_torch(verbose=True)
        print("ColossalAI initialization successful!")
    except Exception as e:
        print(f"ColossalAI initialization failed: {e}")
        return 1

    # Your training code here
    # ...

    return 0

if __name__ == "__main__":
    exit(main())
```

### Command-Line Diagnostics

```python
# Create a diagnostic script: diagnose_k8s.py
from colossalai.utils.k8s_distributed import diagnose_distributed_issues
import json

def main():
    print("Running Kubernetes distributed training diagnostics...")
    diagnosis = diagnose_distributed_issues()

    print("\\n" + "="*50)
    print("DIAGNOSIS RESULTS")
    print("="*50)

    for check, status in diagnosis.items():
        if check == "recommendations":
            continue
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{check:.<30} {status_str}")

    if diagnosis["recommendations"]:
        print("\\n" + "="*50)
        print("RECOMMENDATIONS")
        print("="*50)
        for i, rec in enumerate(diagnosis["recommendations"], 1):
            print(f"{i}. {rec}")

    print("\\n" + json.dumps(diagnosis, indent=2))

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Training Hangs at Initialization
**Symptoms**: Process seems to hang after "Initializing distributed process group"

**Solutions**:
- Check if all pods are running: `kubectl get pods -l app=colossalai-training`
- Verify master service: `kubectl get svc training-master-service`
- Check DNS resolution: `kubectl exec -it <pod-name> -- nslookup training-master-service.default.svc.cluster.local`
- Increase timeout: Set `COLOSSALAI_DIST_TIMEOUT=3600` (1 hour)

#### 2. DNS Resolution Failures
**Symptoms**: "DNS resolution failed" errors

**Solutions**:
- Ensure headless service is created correctly
- Check service selector matches pod labels
- Verify namespace is correct in service DNS name

#### 3. Port Connection Issues
**Symptoms**: "Cannot connect to master_addr:master_port"

**Solutions**:
- Verify master pod is running and healthy
- Check if port 29500 is open and not conflicting
- Ensure no firewall rules blocking communication

#### 4. Environment Variable Issues
**Symptoms**: "Missing required environment variables"

**Solutions**:
- Double-check all required environment variables are set
- Verify NODE_RANK is correctly derived from job completion index
- Check WORLD_SIZE matches total number of processes (nodes × GPUs per node)

### Debug Mode

Enable debug mode for detailed logging:

```bash
# In your pod environment
export NCCL_DEBUG=INFO
export COLOSSALAI_DEBUG=1
export DEBUG_DISTRIBUTED=1

# Check logs
kubectl logs <pod-name> -f
```

### Testing Your Setup

Before running the full training, test the distributed setup:

```python
# test_distributed.py
import torch
import colossalai

def test_distributed_setup():
    try:
        colossalai.launch_from_torch(verbose=True)

        # Basic distributed test
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        print(f"Process {rank}/{world_size} initialized successfully!")

        # Test all-reduce operation
        tensor = torch.ones(1).cuda() * rank
        torch.distributed.all_reduce(tensor)
        expected = sum(range(world_size))

        if tensor.item() == expected:
            print(f"✓ All-reduce test passed: {tensor.item()} == {expected}")
        else:
            print(f"✗ All-reduce test failed: {tensor.item()} != {expected}")

        return True
    except Exception as e:
        print(f"Distributed setup test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_distributed_setup()
    exit(0 if success else 1)
```

## Monitoring and Logging

### View Job Status
```bash
# Check job status
kubectl describe job opensora-multinode-training

# Check pod status
kubectl get pods -l job-name=opensora-multinode-training

# View logs from all pods
kubectl logs -l job-name=opensora-multinode-training -f --prefix=true
```

### Monitor Resource Usage
```bash
# Check GPU usage
kubectl top pods -l job-name=opensora-multinode-training

# Check detailed resource usage
kubectl describe pods -l job-name=opensora-multinode-training
```

## Performance Considerations

1. **Shared Memory**: Ensure adequate `/dev/shm` size for large models
2. **Network Bandwidth**: Verify inter-node network performance
3. **Storage**: Use fast, shared storage for datasets
4. **Node Affinity**: Consider placing pods on high-bandwidth connected nodes

## Advanced Configuration

For production deployments, consider:

1. **Resource Requests/Limits**: Set appropriate CPU/memory/GPU limits
2. **Node Selectors**: Target specific GPU-enabled nodes
3. **Tolerations**: Handle node taints appropriately
4. **Priority Classes**: Set job priority for resource contention
5. **Pod Disruption Budgets**: Protect against voluntary disruptions

This enhanced setup should resolve the multi-node training hanging issue and provide a robust foundation for distributed training in Kubernetes environments.
