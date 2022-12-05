import torch
import transformers
from packaging import version
from torch.utils.data import SequentialSampler
from transformers import BertConfig, BertForSequenceClassification

from .registry import non_distributed_component_funcs


def get_bert_data_loader(
        n_class,
        batch_size,
        total_samples,
        sequence_length,
        device=torch.device('cpu:0'),
        is_distrbuted=False,
):
    train_data = torch.randint(
        low=0,
        high=n_class,
        size=(total_samples, sequence_length),
        device=device,
        dtype=torch.long,
    )
    train_label = torch.randint(low=0, high=2, size=(total_samples,), device=device, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    if is_distrbuted:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    return train_loader


@non_distributed_component_funcs.register(name='bert')
def get_training_components():
    hidden_dim = 8
    num_head = 4
    sequence_length = 12
    num_layer = 2
    vocab_size = 32

    def bert_model_builder(checkpoint: bool = False):
        config = BertConfig(vocab_size=vocab_size,
                            gradient_checkpointing=checkpoint,
                            hidden_size=hidden_dim,
                            intermediate_size=hidden_dim * 4,
                            num_attention_heads=num_head,
                            max_position_embeddings=sequence_length,
                            num_hidden_layers=num_layer,
                            hidden_dropout_prob=0.,
                            attention_probs_dropout_prob=0.)
        print('building BertForSequenceClassification model')

        # adapting huggingface BertForSequenceClassification for single unitest calling interface
        class ModelAaptor(BertForSequenceClassification):

            def forward(self, input_ids, labels):
                """
                inputs: data, label
                outputs: loss
                """
                return super().forward(input_ids=input_ids, labels=labels)[0]

        model = ModelAaptor(config)
        if checkpoint and version.parse(transformers.__version__) >= version.parse("4.11.0"):
            model.gradient_checkpointing_enable()

        return model

    is_distrbuted = torch.distributed.is_initialized()
    trainloader = get_bert_data_loader(n_class=vocab_size,
                                       batch_size=2,
                                       total_samples=10000,
                                       sequence_length=sequence_length,
                                       is_distrbuted=is_distrbuted)
    testloader = get_bert_data_loader(n_class=vocab_size,
                                      batch_size=2,
                                      total_samples=10000,
                                      sequence_length=sequence_length,
                                      is_distrbuted=is_distrbuted)

    criterion = None
    return bert_model_builder, trainloader, testloader, torch.optim.Adam, criterion
