import torch


def calc_acc(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    correct = torch.sum(targets == preds)
    return correct
