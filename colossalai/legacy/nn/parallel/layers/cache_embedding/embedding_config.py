import torch


class TablewiseEmbeddingBagConfig:
    """
    example:
    def prepare_tablewise_config(args, cache_ratio, ...):
        embedding_bag_config_list: List[TablewiseEmbeddingBagConfig] = []
        ...
        return embedding_bag_config_list
    """

    def __init__(
        self,
        num_embeddings: int,
        cuda_row_num: int,
        assigned_rank: int = 0,
        buffer_size=50_000,
        ids_freq_mapping=None,
        initial_weight: torch.tensor = None,
        name: str = "",
    ):
        self.num_embeddings = num_embeddings
        self.cuda_row_num = cuda_row_num
        self.assigned_rank = assigned_rank
        self.buffer_size = buffer_size
        self.ids_freq_mapping = ids_freq_mapping
        self.initial_weight = initial_weight
        self.name = name
