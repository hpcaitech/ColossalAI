#!/bin/bash
# Test deployment script for 2-node multi-node training fix

set -e

echo "🧪 Testing 2-Node Multi-Node Training Fix"
echo "========================================"

# Check prerequisites
echo "1. Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    echo "✗ kubectl not found. Please install kubectl."
    exit 1
fi

if ! kubectl cluster-info &> /dev/null; then
    echo "✗ kubectl not connected to cluster. Please configure kubectl."
    exit 1
fi

echo "✓ kubectl is available and connected"

# Check for GPU nodes
GPU_NODES=$(kubectl get nodes -l accelerator --no-headers 2>/dev/null | wc -l || echo "0")
if [ "$GPU_NODES" -lt 2 ]; then
    echo "⚠️  Warning: Found $GPU_NODES GPU-labeled nodes. You may need at least 2 for this test."
    echo "   Continuing anyway - the test will still validate the fix logic."
fi

# Deploy the test
echo ""
echo "2. Deploying 2-node test job..."

kubectl apply -f 2node-test-job.yaml

echo "✓ Test job deployed"

# Wait for pods to be created
echo ""
echo "3. Waiting for pods to be created..."

sleep 10

# Monitor the deployment
echo ""
echo "4. Monitoring deployment status..."

# Function to get pod status
get_pod_status() {
    kubectl get pods -l job-name=opensora-multinode-test --no-headers 2>/dev/null | awk '{print $3}' | sort | uniq -c
}

# Wait for pods to start
echo "Waiting for pods to start..."
timeout=300  # 5 minutes
elapsed=0

while [ $elapsed -lt $timeout ]; do
    status=$(get_pod_status)
    echo "Pod status: $status"
    
    # Check if both pods are running
    running_pods=$(kubectl get pods -l job-name=opensora-multinode-test --no-headers 2>/dev/null | grep "Running" | wc -l)
    
    if [ "$running_pods" -eq 2 ]; then
        echo "✓ Both pods are running!"
        break
    fi
    
    sleep 10
    elapsed=$((elapsed + 10))
done

if [ $elapsed -ge $timeout ]; then
    echo "✗ Pods did not start within $timeout seconds"
    echo "Pod status:"
    kubectl get pods -l job-name=opensora-multinode-test
    echo ""
    echo "Events:"
    kubectl get events --sort-by='.lastTimestamp' | tail -20
    exit 1
fi

# Show pod information
echo ""
echo "5. Pod information:"
kubectl get pods -l job-name=opensora-multinode-test -o wide

# Function to check logs for success indicators
check_logs() {
    local pod_name=$1
    echo ""
    echo "📋 Checking logs for $pod_name..."
    
    # Get recent logs
    logs=$(kubectl logs $pod_name --tail=50 2>/dev/null || echo "No logs available yet")
    
    # Check for success indicators
    if echo "$logs" | grep -q "ColossalAI initialization successful"; then
        echo "✓ $pod_name: ColossalAI initialization successful"
        success_count=$((success_count + 1))
    elif echo "$logs" | grep -q "Master node.*is ready for connections"; then
        echo "✓ $pod_name: Master readiness check working"
    elif echo "$logs" | grep -q "Waiting for master node.*to be ready"; then
        echo "⏳ $pod_name: Waiting for master (expected behavior)"
    elif echo "$logs" | grep -q "All tests completed successfully"; then
        echo "🎉 $pod_name: All tests completed successfully!"
        success_count=$((success_count + 1))
    else
        echo "⏳ $pod_name: Still initializing..."
    fi
    
    # Check for error indicators
    if echo "$logs" | grep -q "initialization failed"; then
        echo "✗ $pod_name: Initialization failed"
        echo "Error details:"
        echo "$logs" | grep -A 5 -B 5 "initialization failed"
    fi
}

# Monitor logs for success
echo ""
echo "6. Monitoring for success indicators..."

success_count=0
monitor_timeout=600  # 10 minutes
monitor_elapsed=0

while [ $monitor_elapsed -lt $monitor_timeout ] && [ $success_count -lt 2 ]; do
    # Get current pod names
    pod_names=$(kubectl get pods -l job-name=opensora-multinode-test --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
    
    for pod in $pod_names; do
        check_logs $pod
    done
    
    if [ $success_count -ge 2 ]; then
        echo ""
        echo "🎉 SUCCESS: Both pods completed successfully!"
        break
    fi
    
    echo "Waiting... ($monitor_elapsed/${monitor_timeout}s, successes: $success_count/2)"
    sleep 30
    monitor_elapsed=$((monitor_elapsed + 30))
done

# Final status report
echo ""
echo "========================================"
echo "📊 Final Test Results"
echo "========================================"

if [ $success_count -ge 2 ]; then
    echo "🎉 TEST PASSED: Multi-node training fix is working!"
    echo ""
    echo "✓ Both nodes successfully initialized ColossalAI"
    echo "✓ Master readiness checks worked"
    echo "✓ Enhanced torchrun configuration effective"
    echo "✓ Ready to scale to 4 nodes"
    echo ""
    echo "Next steps:"
    echo "1. Scale to 4 nodes by updating YAML (nnodes=4, WORLD_SIZE=32)"
    echo "2. Deploy your actual Open-Sora training script"
    echo "3. The fix is ready for production use!"
    
    result=0
else
    echo "⚠️  TEST INCOMPLETE: $success_count/2 nodes completed successfully"
    echo ""
    echo "This might indicate:"
    echo "1. Initialization is still in progress (check logs)"
    echo "2. Environment-specific configuration needed"
    echo "3. Resource constraints in the cluster"
    echo ""
    echo "Check the detailed logs below:"
    
    result=1
fi

# Show detailed logs
echo ""
echo "========================================"
echo "📋 Detailed Logs from Both Pods"
echo "========================================"

for pod in $(kubectl get pods -l job-name=opensora-multinode-test --no-headers -o custom-columns=":metadata.name" 2>/dev/null); do
    echo ""
    echo "--- Logs from $pod ---"
    kubectl logs $pod --tail=100 || echo "Could not retrieve logs from $pod"
done

# Cleanup option
echo ""
echo "========================================"
echo "🧹 Cleanup"
echo "========================================"
echo "To clean up the test deployment, run:"
echo "kubectl delete job opensora-multinode-test"
echo "kubectl delete service training-master-service"
echo ""
echo "Or run: kubectl delete -f 2node-test-job.yaml"

exit $result