import torch

__all__ = ["LossForPretraining"]


class LossForPretraining(torch.nn.Module):
    def __init__(self, vocab_size):
        super(LossForPretraining, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, masked_lm_labels, next_sentence_labels=None):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        # next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss  # + next_sentence_loss
        return total_loss
