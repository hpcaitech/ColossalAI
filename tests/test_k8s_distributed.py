#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
Test suite for Kubernetes distributed training enhancements.

This module tests the enhanced distributed training functionality
that addresses multi-node training hanging issues in K8s environments.
"""

import os
import socket
import threading
from unittest.mock import MagicMock, patch

import pytest
import torch.distributed as dist

from colossalai.initialize import _get_distributed_timeout, _wait_for_master_ready, launch_from_torch
from colossalai.utils.k8s_distributed import (
    diagnose_distributed_issues,
    generate_torchrun_command,
    setup_k8s_networking,
    validate_k8s_environment,
)


class TestK8sDistributedEnhancements:
    """Test class for Kubernetes distributed training enhancements."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clean up any existing distributed state
        if dist.is_initialized():
            dist.destroy_process_group()

        # Clear environment variables that might interfere with tests
        test_env_vars = [
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "NCCL_DEBUG",
            "GLOO_SOCKET_IFNAME",
            "NCCL_SOCKET_IFNAME",
            "COLOSSALAI_DIST_TIMEOUT",
            "COLOSSALAI_MASTER_READY_TIMEOUT",
        ]
        for var in test_env_vars:
            if var in os.environ:
                del os.environ[var]

    def teardown_method(self):
        """Clean up after each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_wait_for_master_ready_success(self):
        """Test successful master readiness check."""
        # Create a mock server that accepts connections
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("localhost", 0))
        port = server_socket.getsockname()[1]
        server_socket.listen(1)

        def server_thread():
            try:
                conn, addr = server_socket.accept()
                conn.close()
            except:
                pass
            finally:
                server_socket.close()

        # Start server in background
        thread = threading.Thread(target=server_thread)
        thread.daemon = True
        thread.start()

        # Test connection readiness
        result = _wait_for_master_ready("localhost", port, timeout=10, retry_interval=1)
        assert result is True

        thread.join(timeout=1)

    def test_wait_for_master_ready_timeout(self):
        """Test master readiness check timeout."""
        # Use a port that won't accept connections
        result = _wait_for_master_ready("localhost", 65432, timeout=2, retry_interval=1)
        assert result is False

    def test_get_distributed_timeout_default(self):
        """Test default timeout configuration."""
        timeout = _get_distributed_timeout()
        assert timeout.seconds == 1800  # 30 minutes default

    def test_get_distributed_timeout_custom(self):
        """Test custom timeout configuration."""
        os.environ["COLOSSALAI_DIST_TIMEOUT"] = "3600"
        timeout = _get_distributed_timeout()
        assert timeout.seconds == 3600  # 1 hour

    def test_validate_k8s_environment_missing_vars(self):
        """Test environment validation with missing variables."""
        with pytest.raises(RuntimeError) as exc_info:
            validate_k8s_environment()

        assert "Missing essential environment variables" in str(exc_info.value)
        assert "MASTER_ADDR" in str(exc_info.value)

    def test_validate_k8s_environment_complete(self):
        """Test environment validation with all required variables."""
        # Set required environment variables
        test_env = {
            "MASTER_ADDR": "test-master.example.com",
            "MASTER_PORT": "29500",
            "WORLD_SIZE": "8",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "NODE_RANK": "0",
        }

        for key, value in test_env.items():
            os.environ[key] = value

        env_vars = validate_k8s_environment()

        # Check that all variables are returned
        for key in test_env:
            assert key in env_vars
            assert env_vars[key] == test_env[key]

    def test_setup_k8s_networking(self):
        """Test K8s networking setup."""
        setup_k8s_networking()

        # Check that networking environment variables are set
        assert os.environ.get("NCCL_IB_DISABLE") == "1"
        assert os.environ.get("NCCL_SOCKET_IFNAME") == "eth0"
        assert os.environ.get("GLOO_SOCKET_IFNAME") == "eth0"
        assert "NCCL_DEBUG" in os.environ

    def test_generate_torchrun_command(self):
        """Test torchrun command generation."""
        # Set environment variables
        os.environ.update(
            {
                "NNODES": "4",
                "NPROC_PER_NODE": "8",
                "NODE_RANK": "0",
                "MASTER_ADDR": "master.example.com",
                "MASTER_PORT": "29500",
            }
        )

        script_path = "train.py"
        script_args = ["--config", "config.yaml"]

        command = generate_torchrun_command(script_path, script_args)

        # Check that command contains expected elements
        assert "torchrun" in command
        assert "--nnodes=4" in command
        assert "--nproc_per_node=8" in command
        assert "--rdzv_backend=c10d" in command
        assert "--rdzv_endpoint=master.example.com:29500" in command
        assert "train.py" in command
        assert "--config" in command
        assert "config.yaml" in command

    def test_launch_from_torch_missing_env_vars(self):
        """Test launch_from_torch with missing environment variables."""
        with pytest.raises(RuntimeError) as exc_info:
            launch_from_torch()

        assert "Missing required environment variables" in str(exc_info.value)
        assert "torchrun" in str(exc_info.value)  # Should mention enhanced torchrun command

    def test_launch_from_torch_invalid_values(self):
        """Test launch_from_torch with invalid environment variable values."""
        # Set environment variables with invalid values
        os.environ.update(
            {
                "RANK": "not_a_number",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "4",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29500",
            }
        )

        with pytest.raises(RuntimeError) as exc_info:
            launch_from_torch()

        assert "Invalid environment variable value" in str(exc_info.value)

    def test_launch_from_torch_validation_checks(self):
        """Test launch_from_torch parameter validation."""
        # Test RANK >= WORLD_SIZE
        os.environ.update(
            {"RANK": "4", "LOCAL_RANK": "0", "WORLD_SIZE": "4", "MASTER_ADDR": "localhost", "MASTER_PORT": "29500"}
        )

        with pytest.raises(RuntimeError) as exc_info:
            launch_from_torch()

        assert "RANK (4) must be less than WORLD_SIZE (4)" in str(exc_info.value)

        # Test invalid port
        os.environ.update({"RANK": "0", "MASTER_PORT": "99999"})  # Invalid port

        with pytest.raises(RuntimeError) as exc_info:
            launch_from_torch()

        assert "MASTER_PORT (99999) must be between 1024 and 65535" in str(exc_info.value)

    @patch("torch.distributed.init_process_group")
    @patch("colossalai.accelerator.get_accelerator")
    def test_launch_from_torch_success(self, mock_accelerator, mock_init_pg):
        """Test successful launch_from_torch execution."""
        # Mock accelerator
        mock_acc = MagicMock()
        mock_acc.communication_backend = "nccl"
        mock_acc.support_set_device = True
        mock_accelerator.return_value = mock_acc

        # Set valid environment variables
        os.environ.update(
            {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2", "MASTER_ADDR": "localhost", "MASTER_PORT": "29500"}
        )

        # Mock successful process group initialization
        mock_init_pg.return_value = None

        # Should not raise any exceptions
        launch_from_torch(verbose=False)

        # Verify that init_process_group was called with timeout
        mock_init_pg.assert_called_once()
        call_args = mock_init_pg.call_args
        assert "timeout" in call_args.kwargs

    def test_diagnose_distributed_issues(self):
        """Test distributed training diagnostics."""
        # Set some environment variables for testing
        os.environ.update(
            {"MASTER_ADDR": "localhost", "MASTER_PORT": "29500", "WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}
        )

        diagnosis = diagnose_distributed_issues()

        # Check that diagnosis returns expected structure
        assert "network_connectivity" in diagnosis
        assert "dns_resolution" in diagnosis
        assert "port_availability" in diagnosis
        assert "environment_variables" in diagnosis
        assert "recommendations" in diagnosis

        # Environment variables should pass
        assert diagnosis["environment_variables"] is True
        assert isinstance(diagnosis["recommendations"], list)


class TestK8sYamlGeneration:
    """Test YAML generation functions."""

    def test_create_k8s_headless_service_yaml(self):
        """Test headless service YAML generation."""
        from colossalai.utils.k8s_distributed import create_k8s_headless_service_yaml

        yaml_content = create_k8s_headless_service_yaml(
            service_name="test-service", namespace="test-ns", port=12345, app_label="test-app"
        )

        # Check that YAML contains expected content
        assert "name: test-service" in yaml_content
        assert "namespace: test-ns" in yaml_content
        assert "port: 12345" in yaml_content
        assert "app: test-app" in yaml_content
        assert "clusterIP: None" in yaml_content

    def test_create_k8s_job_yaml(self):
        """Test training job YAML generation."""
        from colossalai.utils.k8s_distributed import create_k8s_job_yaml

        yaml_content = create_k8s_job_yaml(
            job_name="test-job",
            namespace="test-ns",
            image="test:latest",
            num_nodes=2,
            gpus_per_node=4,
            script_command=["python", "test.py"],
        )

        # Check that YAML contains expected content
        assert "name: test-job" in yaml_content
        assert "namespace: test-ns" in yaml_content
        assert "image: test:latest" in yaml_content
        assert "parallelism: 2" in yaml_content
        assert "completions: 2" in yaml_content
        assert "nvidia.com/gpu: 4" in yaml_content
        assert "python test.py" in yaml_content


@pytest.mark.skip(reason="Requires actual distributed environment")
class TestDistributedIntegration:
    """Integration tests that require actual distributed setup."""

    def test_full_distributed_initialization(self):
        """Test complete distributed initialization flow."""
        # This would require actual multi-process setup
        # Skip for now but can be used for manual testing


if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v", "-s"])
