import torch
import transformers
from packaging import version
from transformers import AlbertConfig, AlbertForSequenceClassification

from .bert import get_bert_data_loader
from .registry import non_distributed_component_funcs


@non_distributed_component_funcs.register(name='albert')
def get_training_components():
    hidden_dim = 8
    num_head = 4
    sequence_length = 12
    num_layer = 2
    vocab_size = 32

    def bert_model_builder(checkpoint: bool = False):
        config = AlbertConfig(vocab_size=vocab_size,
                              gradient_checkpointing=checkpoint,
                              hidden_size=hidden_dim,
                              intermediate_size=hidden_dim * 4,
                              num_attention_heads=num_head,
                              max_position_embeddings=sequence_length,
                              num_hidden_layers=num_layer,
                              hidden_dropout_prob=0.,
                              attention_probs_dropout_prob=0.)
        print('building AlbertForSequenceClassification model')

        # adapting huggingface BertForSequenceClassification for single unitest calling interface
        class ModelAaptor(AlbertForSequenceClassification):

            def forward(self, input_ids, labels):
                """
                inputs: data, label
                outputs: loss
                """
                return super().forward(input_ids=input_ids, labels=labels)[0]

        model = ModelAaptor(config)
        # if checkpoint and version.parse(transformers.__version__) >= version.parse("4.11.0"):
        #     model.gradient_checkpointing_enable()

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
