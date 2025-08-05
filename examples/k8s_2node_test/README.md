# Minimal 2-Node Test Setup for Multi-Node Training Fix

This directory contains a minimal setup to test the multi-node training fix before scaling to full 4-node deployment.

## Quick Test Commands

### For the Original Issue Reporter (@ltm920716)

Replace your problematic command:
```bash
# OLD (gets stuck):
torchrun --nnodes 4 --nproc_per_node 8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node-rank $NODE_RANK scripts/diffusion/train.py configs/diffusion/train/demo.py --dataset.data-path modified_data.csv
```

With this enhanced version (start with 2 nodes):
```bash
# NEW (should work):
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=$NODE_RANK \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=opensora-test-2node \
  --rdzv_conf="timeout=1800,read_timeout=120" \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/diffusion/train.py configs/diffusion/train/demo.py \
  --dataset.data-path modified_data.csv
```

## Environment Variables

Set these in your Kubernetes deployment:

### Essential Variables
```yaml
env:
  - name: MASTER_ADDR
    value: "training-master-service.default.svc.cluster.local"  # or your service name
  - name: MASTER_PORT
    value: "29500"
  - name: WORLD_SIZE
    value: "16"  # 2 nodes × 8 GPUs
  - name: NODE_RANK
    valueFrom:
      fieldRef:
        fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
  
  # Enhanced timeout configuration
  - name: COLOSSALAI_DIST_TIMEOUT
    value: "1800"  # 30 minutes for init
  - name: COLOSSALAI_MASTER_READY_TIMEOUT
    value: "600"   # 10 minutes for master readiness
  
  # K8s networking
  - name: NCCL_SOCKET_IFNAME
    value: "eth0"
  - name: GLOO_SOCKET_IFNAME  
    value: "eth0"
  - name: NCCL_IB_DISABLE
    value: "1"
  - name: NCCL_DEBUG
    value: "WARN"  # Use "INFO" for debugging
```

## Expected Behavior Changes

### Before Fix (Hanging):
```
Process 0: Starting distributed training...
Process 1: Starting distributed training...
[Both processes hang here indefinitely]
```

### After Fix (Working):
```
Process 0: Starting distributed training with configuration:
Process 0:   RANK: 0
Process 0:   WORLD_SIZE: 16
Process 0:   MASTER_ADDR: training-master-service.default.svc.cluster.local
Process 0: Initializing distributed process group: rank=0, world_size=16, timeout=0:30:00
Process 0: ✓ ColossalAI initialization successful!

Process 8: Rank 8: Waiting for master node to be ready...
Process 8: Master node training-master-service.default.svc.cluster.local:29500 is ready for connections
Process 8: Initializing distributed process group: rank=8, world_size=16, timeout=0:30:00
Process 8: ✓ ColossalAI initialization successful!
```

## Kubernetes YAML for 2-Node Test

Save as `2node-test-job.yaml`:

```yaml
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
---
apiVersion: batch/v1
kind: Job
metadata:
  name: opensora-2node-test
  namespace: default
spec:
  parallelism: 2
  completions: 2
  completionMode: Indexed
  template:
    metadata:
      labels:
        app: colossalai-training
        role: master  # Both pods can be master for simplicity in 2-node test
    spec:
      restartPolicy: Never
      containers:
      - name: training
        image: your-opensora-image:latest  # Replace with your image
        command:
        - /bin/bash
        - -c
        - |
          # Stagger startup to avoid race conditions
          sleep $((JOB_COMPLETION_INDEX * 30 + 30))
          
          # Set rank based on completion index
          export RANK=$((JOB_COMPLETION_INDEX * 8))
          export LOCAL_RANK=0
          
          echo "Node $JOB_COMPLETION_INDEX starting with RANK=$RANK"
          
          # Enhanced torchrun command
          torchrun \
            --nnodes=2 \
            --nproc_per_node=8 \
            --node_rank=$JOB_COMPLETION_INDEX \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            --rdzv_id=opensora-2node-test \
            --rdzv_conf="timeout=1800,read_timeout=120" \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            scripts/diffusion/train.py configs/diffusion/train/demo.py \
            --dataset.data-path modified_data.csv
        env:
        - name: MASTER_ADDR
          value: "training-master-service.default.svc.cluster.local"
        - name: MASTER_PORT
          value: "29500"
        - name: WORLD_SIZE
          value: "16"
        - name: NODE_RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        - name: COLOSSALAI_DIST_TIMEOUT
          value: "1800"
        - name: COLOSSALAI_MASTER_READY_TIMEOUT
          value: "600"
        - name: NCCL_SOCKET_IFNAME
          value: "eth0"
        - name: GLOO_SOCKET_IFNAME
          value: "eth0"
        - name: NCCL_IB_DISABLE
          value: "1"
        - name: NCCL_DEBUG
          value: "INFO"  # Enable detailed logging for testing
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
          sizeLimit: 16Gi
```

## Testing Steps

1. **Deploy the test:**
   ```bash
   kubectl apply -f 2node-test-job.yaml
   ```

2. **Monitor the pods:**
   ```bash
   kubectl get pods -l job-name=opensora-2node-test -w
   ```

3. **Check logs:**
   ```bash
   # Watch logs from both pods
   kubectl logs -l job-name=opensora-2node-test -f --prefix=true
   
   # Or individual pods
   kubectl logs opensora-2node-test-0-xxxxx -f
   kubectl logs opensora-2node-test-1-xxxxx -f
   ```

4. **Look for success indicators:**
   - "Master node X:Y is ready for connections"
   - "ColossalAI initialization successful!"
   - Training actually starts (loss values, etc.)

5. **If successful, scale to 4 nodes:**
   - Change `parallelism: 4`, `completions: 4`
   - Update `WORLD_SIZE: "32"`
   - Update `--nnodes=4` in torchrun command

## Troubleshooting

### If it still hangs:
1. Check service DNS resolution:
   ```bash
   kubectl exec -it opensora-2node-test-0-xxxxx -- nslookup training-master-service.default.svc.cluster.local
   ```

2. Check port connectivity:
   ```bash
   kubectl exec -it opensora-2node-test-1-xxxxx -- telnet training-master-service.default.svc.cluster.local 29500
   ```

3. Enable debug mode:
   ```bash
   # Add to pod environment
   - name: DEBUG_DISTRIBUTED
     value: "1"
   - name: NCCL_DEBUG
     value: "INFO" 
   ```

### Common Issues:
- **DNS resolution fails**: Check service name and namespace
- **Port not accessible**: Verify both pods are running
- **Timeout too short**: Increase `COLOSSALAI_DIST_TIMEOUT`

## Success Criteria

✅ **2-node test passes if:**
- Both pods start without hanging
- You see "ColossalAI initialization successful!" in logs
- Training loop actually begins
- No indefinite waiting at initialization

Once 2-node test passes, you can confidently scale to 4 nodes with the same enhanced configuration.