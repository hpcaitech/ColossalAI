import torch


def run_fwd_bwd(model, data, label, criterion, enable_autocast=False, use_init_ctx=False):
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        if criterion:
            y = model(data)
            loss = criterion(y, label)
        else:
            loss = model(data, label)
        loss = loss.float()
    if use_init_ctx:
        model.backward(loss)
    else:
        loss.backward()
