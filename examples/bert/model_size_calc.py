def get_model_numel(model):
    numel = 0
    param_cnt = 0
    for _, param in model.named_parameters(recurse=True):
        numel += param.numel()
        param_cnt += 1
    return numel, param_cnt