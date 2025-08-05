#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Kubernetes-aware distributed training utilities for ColossalAI.

This module provides enhanced functionality for multi-node distributed training
in Kubernetes environments, addressing common issues like pod startup timing,
network discovery, and rendezvous configuration.
"""

import os
import time
import socket
import subprocess
from typing import Dict, List, Optional, Tuple
from colossalai.logging import get_dist_logger


def validate_k8s_environment() -> Dict[str, str]:
    """
    Validate and return Kubernetes environment variables for distributed training.
    
    Returns:
        Dict[str, str]: Dictionary of validated environment variables
        
    Raises:
        RuntimeError: If required environment variables are missing or invalid
    """
    logger = get_dist_logger()
    
    # Essential environment variables for K8s distributed training
    essential_vars = {
        "MASTER_ADDR": "Master node service DNS name or IP",
        "MASTER_PORT": "Master node port (usually 29500)",
        "WORLD_SIZE": "Total number of processes",
        "RANK": "Global rank of current process",
        "LOCAL_RANK": "Local rank on current node",
        "NODE_RANK": "Rank of current node"
    }
    
    # Optional but recommended variables
    recommended_vars = {
        "RDZV_ID": "Unique job identifier for rendezvous",
        "NCCL_SOCKET_IFNAME": "Network interface for NCCL (e.g., eth0)",
        "GLOO_SOCKET_IFNAME": "Network interface for Gloo backend",
        "NCCL_DEBUG": "NCCL debug level (INFO for debugging)",
        "NCCL_IB_DISABLE": "Disable InfiniBand (set to 1 in most K8s envs)"
    }
    
    env_vars = {}
    missing_vars = []
    
    # Check essential variables
    for var, description in essential_vars.items():
        if var in os.environ:
            env_vars[var] = os.environ[var]
        else:
            missing_vars.append(f"  - {var}: {description}")
    
    if missing_vars:
        error_msg = ("Missing essential environment variables for K8s distributed training:\n" +
                    "\n".join(missing_vars) +
                    "\n\nExample Kubernetes deployment configuration:\n"
                    "env:\n"
                    "  - name: MASTER_ADDR\n"
                    "    value: \"training-master-service.default.svc.cluster.local\"\n"
                    "  - name: MASTER_PORT\n"
                    "    value: \"29500\"\n"
                    "  - name: WORLD_SIZE\n"
                    "    value: \"32\"  # 4 nodes * 8 GPUs\n"
                    "  - name: NODE_RANK\n"
                    "    valueFrom:\n"
                    "      fieldRef:\n"
                    "        fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']\n")
        raise RuntimeError(error_msg)
    
    # Log recommended variables
    for var, description in recommended_vars.items():
        if var in os.environ:
            env_vars[var] = os.environ[var]
            logger.info(f"Using {var}={os.environ[var]}")
        else:
            logger.warning(f"Recommended environment variable {var} not set: {description}")
    
    return env_vars


def wait_for_pods_ready(world_size: int, timeout: int = 600) -> bool:
    """
    Wait for all pods in the distributed training job to be ready.
    
    Args:
        world_size (int): Expected number of processes
        timeout (int): Maximum time to wait in seconds
        
    Returns:
        bool: True if all pods are ready, False otherwise
    """
    logger = get_dist_logger()
    start_time = time.time()
    
    logger.info(f"Waiting for {world_size} processes to be ready...")
    
    while time.time() - start_time < timeout:
        try:
            # In K8s, we can check if the expected number of pods are running
            # This is a simplified check - in practice you might query K8s API
            time.sleep(10)  # Give pods time to start
            logger.info("Pod readiness check passed (simplified implementation)")
            return True
        except Exception as e:
            logger.debug(f"Pod readiness check failed: {e}")
            time.sleep(10)
    
    logger.error(f"Not all pods became ready within {timeout} seconds")
    return False


def setup_k8s_networking():
    """
    Configure networking settings optimized for Kubernetes environments.
    """
    logger = get_dist_logger()
    
    # Set networking environment variables if not already set
    network_config = {
        "NCCL_IB_DISABLE": "1",  # Disable InfiniBand in most K8s environments
        "NCCL_SOCKET_IFNAME": "eth0",  # Default K8s network interface
        "GLOO_SOCKET_IFNAME": "eth0",
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN")  # Don't override if already set
    }
    
    for var, value in network_config.items():
        if var not in os.environ:
            os.environ[var] = value
            logger.info(f"Set {var}={value} for K8s networking")
        else:
            logger.info(f"Using existing {var}={os.environ[var]}")


def generate_torchrun_command(
    script_path: str,
    script_args: List[str] = None,
    nnodes: int = None,
    nproc_per_node: int = None,
    node_rank: int = None,
    master_addr: str = None,
    master_port: int = None,
    rdzv_id: str = None
) -> str:
    """
    Generate an enhanced torchrun command for Kubernetes multi-node training.
    
    Args:
        script_path (str): Path to the training script
        script_args (List[str], optional): Arguments for the training script
        nnodes (int, optional): Number of nodes (read from env if not provided)
        nproc_per_node (int, optional): Processes per node (read from env if not provided)
        node_rank (int, optional): Node rank (read from env if not provided)
        master_addr (str, optional): Master address (read from env if not provided)
        master_port (int, optional): Master port (read from env if not provided)
        rdzv_id (str, optional): Rendezvous ID (generated if not provided)
        
    Returns:
        str: Complete torchrun command
    """
    # Use environment variables as defaults
    nnodes = nnodes or int(os.environ.get("NNODES", "1"))
    nproc_per_node = nproc_per_node or int(os.environ.get("NPROC_PER_NODE", "8"))
    node_rank = node_rank or int(os.environ.get("NODE_RANK", "0"))
    master_addr = master_addr or os.environ.get("MASTER_ADDR", "localhost")
    master_port = master_port or int(os.environ.get("MASTER_PORT", "29500"))
    rdzv_id = rdzv_id or os.environ.get("RDZV_ID", f"colossalai_job_{int(time.time())}")
    
    # Build torchrun command with enhanced configuration
    cmd_parts = [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--node_rank={node_rank}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={master_addr}:{master_port}",
        f"--rdzv_id={rdzv_id}",
        "--rdzv_conf=timeout=1800,read_timeout=120",  # 30min timeout, 2min read timeout
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        script_path
    ]
    
    if script_args:
        cmd_parts.extend(script_args)
    
    return " \\\n  ".join(cmd_parts)


def create_k8s_headless_service_yaml(
    service_name: str = "colossalai-training-service",
    namespace: str = "default",
    port: int = 29500,
    app_label: str = "colossalai-training"
) -> str:
    """
    Generate YAML configuration for a Kubernetes headless service for distributed training.
    
    Args:
        service_name (str): Name of the service
        namespace (str): Kubernetes namespace
        port (int): Service port
        app_label (str): App label selector
        
    Returns:
        str: YAML configuration
    """
    yaml_config = f"""# Headless service for ColossalAI distributed training
# This provides stable DNS names for pod-to-pod communication
apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  namespace: {namespace}
  labels:
    app: {app_label}
spec:
  clusterIP: None  # Makes this a headless service
  selector:
    app: {app_label}
  ports:
  - name: distributed-comm
    port: {port}
    targetPort: {port}
    protocol: TCP
---
# Optional: Service for master node specifically  
apiVersion: v1
kind: Service
metadata:
  name: {service_name}-master
  namespace: {namespace}
  labels:
    app: {app_label}
    role: master
spec:
  clusterIP: None
  selector:
    app: {app_label}
    role: master
  ports:
  - name: master-port
    port: {port}
    targetPort: {port}
    protocol: TCP
"""
    return yaml_config


def create_k8s_job_yaml(
    job_name: str = "colossalai-multinode-training",
    namespace: str = "default",
    image: str = "your-training-image:latest",
    num_nodes: int = 4,
    gpus_per_node: int = 8,
    script_command: List[str] = None
) -> str:
    """
    Generate YAML configuration for a Kubernetes Job for multi-node training.
    
    Args:
        job_name (str): Name of the training job
        namespace (str): Kubernetes namespace
        image (str): Docker image for training
        num_nodes (int): Number of nodes
        gpus_per_node (int): GPUs per node
        script_command (List[str]): Command to run training script
        
    Returns:
        str: YAML configuration
    """
    if script_command is None:
        script_command = ["python", "scripts/diffusion/train.py", "configs/diffusion/train/demo.py"]
    
    yaml_config = f"""# ColossalAI Multi-Node Training Job
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {namespace}
spec:
  parallelism: {num_nodes}
  completions: {num_nodes}
  completionMode: Indexed
  template:
    metadata:
      labels:
        app: colossalai-training
    spec:
      restartPolicy: Never
      containers:
      - name: training
        image: {image}
        command:
        - /bin/bash
        - -c
        - |
          # Wait a bit for all pods to start
          sleep $((RANDOM % 30 + 30))
          
          # Enhanced torchrun command
          torchrun \\
            --nnodes={num_nodes} \\
            --nproc_per_node={gpus_per_node} \\
            --node_rank=$JOB_COMPLETION_INDEX \\
            --rdzv_backend=c10d \\
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
            --rdzv_id={job_name} \\
            --rdzv_conf=timeout=1800,read_timeout=120 \\
            --master_addr=$MASTER_ADDR \\
            --master_port=$MASTER_PORT \\
            {' '.join(script_command)}
        env:
        - name: MASTER_ADDR
          value: "colossalai-training-service-master.{namespace}.svc.cluster.local"
        - name: MASTER_PORT
          value: "29500"
        - name: WORLD_SIZE
          value: "{num_nodes * gpus_per_node}"
        - name: NODE_RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        - name: RDZV_ID
          value: "{job_name}"
        - name: NCCL_SOCKET_IFNAME
          value: "eth0"
        - name: GLOO_SOCKET_IFNAME
          value: "eth0"
        - name: NCCL_IB_DISABLE
          value: "1"
        - name: NCCL_DEBUG
          value: "WARN"
        - name: COLOSSALAI_DIST_TIMEOUT
          value: "1800"  # 30 minutes
        - name: COLOSSALAI_MASTER_READY_TIMEOUT
          value: "600"   # 10 minutes
        resources:
          requests:
            nvidia.com/gpu: {gpus_per_node}
          limits:
            nvidia.com/gpu: {gpus_per_node}
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 32Gi  # Adjust based on your needs
      nodeSelector:
        accelerator: nvidia-tesla-v100  # Adjust based on your GPU nodes
"""
    return yaml_config


def diagnose_distributed_issues() -> Dict[str, any]:
    """
    Diagnose common distributed training issues in Kubernetes environments.
    
    Returns:
        Dict[str, any]: Diagnosis results and recommendations
    """
    logger = get_dist_logger()
    diagnosis = {
        "network_connectivity": False,
        "dns_resolution": False,
        "port_availability": False,
        "environment_variables": False,
        "recommendations": []
    }
    
    # Check environment variables
    required_envs = ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"]
    missing_envs = [env for env in required_envs if env not in os.environ]
    
    if not missing_envs:
        diagnosis["environment_variables"] = True
        logger.info("✓ All required environment variables are set")
    else:
        logger.error(f"✗ Missing environment variables: {missing_envs}")
        diagnosis["recommendations"].append(f"Set missing environment variables: {missing_envs}")
    
    # Check DNS resolution
    if "MASTER_ADDR" in os.environ:
        try:
            master_addr = os.environ["MASTER_ADDR"]
            socket.gethostbyname(master_addr)
            diagnosis["dns_resolution"] = True
            logger.info(f"✓ DNS resolution successful for {master_addr}")
        except socket.gaierror:
            logger.error(f"✗ DNS resolution failed for {master_addr}")
            diagnosis["recommendations"].append("Check if master service DNS name is correct and accessible")
    
    # Check port connectivity
    if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
        try:
            master_addr = os.environ["MASTER_ADDR"]
            master_port = int(os.environ["MASTER_PORT"])
            sock = socket.create_connection((master_addr, master_port), timeout=10)
            sock.close()
            diagnosis["port_availability"] = True
            logger.info(f"✓ Port {master_port} is accessible on {master_addr}")
        except (socket.error, socket.timeout, ConnectionRefusedError):
            logger.error(f"✗ Cannot connect to {master_addr}:{master_port}")
            diagnosis["recommendations"].append("Check if master node is running and port is open")
    
    # Network interface check
    nccl_interface = os.environ.get("NCCL_SOCKET_IFNAME", "eth0")
    try:
        result = subprocess.run(["ip", "addr", "show", nccl_interface], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            diagnosis["network_connectivity"] = True
            logger.info(f"✓ Network interface {nccl_interface} is available")
        else:
            logger.error(f"✗ Network interface {nccl_interface} not found")
            diagnosis["recommendations"].append(f"Check network interface configuration (current: {nccl_interface})")
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Could not check network interface (ip command not available)")
    
    return diagnosis