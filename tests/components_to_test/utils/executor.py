import torch


def run_fwd_bwd(model, data, label, criterion, optimizer=None) -> torch.Tensor:
    """run_fwd_bwd
    run fwd and bwd for the model

    Args:
        model (torch.nn.Module): a PyTorch model
        data (torch.Tensor): input data
        label (torch.Tensor): label
        criterion (Optional[Callable]): a function of criterion

    Returns:
        torch.Tensor: loss of fwd
    """
    if criterion:
        y = model(data)
        y = y.float()
        loss = criterion(y, label)
    else:
        loss = model(data, label)

    loss = loss.float()
    if optimizer:
        optimizer.backward(loss)
    else:
        loss.backward()
    return loss
