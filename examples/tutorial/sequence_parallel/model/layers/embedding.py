import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class VocabEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VocabEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        # Allocate weights and initialize.
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, hidden_state):
        output = F.embedding(
            hidden_state,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return output

    def __repr__(self):
        return f"VocabEmbedding(num_embeddings={self.num_embeddings}, " f"embedding_dim={self.embedding_dim})"


class Embedding(nn.Module):
    """Language model embeddings.
    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self, hidden_size, vocab_size, max_sequence_length, embedding_dropout_prob, num_tokentypes):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.num_tokentypes = num_tokentypes

        self.word_embeddings = VocabEmbedding(vocab_size, self.hidden_size)

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    @property
    def word_embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        if tokentype_ids is not None and self.tokentype_embeddings is not None:
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        return embeddings
