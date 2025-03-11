import torch
from coati.distributed.utils import log_probs_from_logits, masked_mean
from torch.testing import assert_close

from colossalai.testing import parameterize
from colossalai.utils import set_seed


@parameterize(
    "test_config",
    [
        {"precision": torch.bfloat16, "device": "npu"},
    ],
)
def run_log_probs_from_logits(test_config):
    torch.set_default_dtype(test_config["precision"])
    set_seed(42)

    # generate input
    logits_cpu = torch.randn(2, 10, 50257)  # (batch, seq_len, vocab_size)
    labels_cpu = torch.randint(0, 50257, (2, 10))

    # to npu
    logits_cpu = logits_cpu
    labels_cpu = labels_cpu
    logits_gpu = logits_cpu.clone().to(device=test_config["device"])
    labels_gpu = labels_cpu.clone().to(device=test_config["device"])

    # fwd
    output_cpu = log_probs_from_logits(logits_cpu, labels_cpu)
    output_gpu = log_probs_from_logits(logits_gpu, labels_gpu).cpu()

    # assert close
    assert_close(
        output_gpu,
        output_cpu,
        rtol=5e-4,
        atol=5e-4,
        # msg=f"NPU/CPU {test_config['precision']} not close"
    )


@parameterize(
    "test_config",
    [
        {"precision": torch.bfloat16, "device": "npu"},
        {"precision": torch.float32, "device": "npu"},
    ],
)
def run_calc_action_log_probs(test_config):
    # same with run_log_probs_from_logits
    pass


@parameterize(
    "test_config",
    [
        {"precision": torch.bfloat16, "device": "npu"},
    ],
)
def run_masked_mean(test_config):
    torch.set_default_dtype(test_config["precision"])
    set_seed(42)

    # init tensor and mask
    tensor = torch.randn(1, 10, 128)  # batch_size, seq_length, hidden_size
    mask = torch.rand(1, 10, 128) > 0.3  # init mask

    tensor_gpu = tensor.to(device=test_config["device"])
    mask_gpu = mask.to(device=test_config["device"])

    # fwd
    cpu_output = masked_mean(tensor, mask, dim=1)
    gpu_output = masked_mean(tensor_gpu, mask_gpu, dim=1).cpu()

    # assert close
    torch.testing.assert_close(cpu_output, gpu_output, atol=1e-2, rtol=1e-2)


def test_util_func():
    run_calc_action_log_probs()
    run_log_probs_from_logits()
    run_masked_mean()


if __name__ == "__main__":
    test_util_func()
