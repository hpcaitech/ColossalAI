"""CPU gradient/update smoke check; this is not the official diffusion objective."""

import torch
import torch.nn.functional as F

from check_llada_tiny import build_tiny_model


def main():
    model = build_tiny_model()
    model.train()
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert parameters
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in parameters.values())

    # Fixed synthetic input and mask exercise the gradient path, not model quality.
    labels = torch.randint(3, model.config.vocab_size, (2, 16))
    masked = torch.zeros_like(labels, dtype=torch.bool)
    masked[:, ::2] = True
    input_ids = labels.clone()
    input_ids[masked] = model.config.mask_token_id
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-3, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)

    print("目标: 固定掩码位置的交叉熵，仅用于梯度和参数更新检查")
    print("掩码位置数量:", masked.sum().item())
    logits = model(input_ids=input_ids, return_dict=True).logits
    assert tuple(logits.shape) == (2, 16, model.config.vocab_size)
    assert torch.isfinite(logits).all().item(), "输出出现 NaN 或无穷大"
    loss = F.cross_entropy(logits[masked], labels[masked])
    assert loss.requires_grad, "loss 没有连接到计算图"
    assert torch.isfinite(loss).item(), "loss 出现 NaN 或无穷大"
    print("LOSS_OK:", loss.item())

    loss.backward()
    missing = [name for name, p in parameters.items() if p.grad is None]
    assert not missing, f"以下参数没有梯度: {missing}"
    invalid = [name for name, p in parameters.items() if not torch.isfinite(p.grad).all().item()]
    assert not invalid, f"以下参数梯度出现 NaN 或无穷大: {invalid}"
    nonzero = [name for name, p in parameters.items() if torch.count_nonzero(p.grad).item() > 0]
    assert nonzero, "所有参数梯度均为零"
    print("具有有限梯度的参数张量:", len(parameters))
    print("具有非零梯度的参数张量:", len(nonzero))
    print("BACKWARD_OK")

    before = {name: p.detach().clone() for name, p in parameters.items()}
    optimizer.step()
    invalid = [name for name, p in parameters.items() if not torch.isfinite(p).all().item()]
    assert not invalid, f"更新后参数出现 NaN 或无穷大: {invalid}"
    changed = [name for name in nonzero if not torch.equal(before[name], parameters[name].detach())]
    assert changed, "优化器未更新任何有非零梯度的参数"
    print("已更新的非零梯度参数张量:", len(changed))
    print("STEP_OK")
    print("范围: CPU / FP32 / tiny / 一步更新；未验证官方损失、数值对齐或分布式训练")


if __name__ == "__main__":
    main()
