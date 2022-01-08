from colossalai.context import ParallelMode
from colossalai.nn.layer import WrappedDropout as Dropout


def moe_sa_args(d_model: int,
                n_heads: int,
                d_kv: int,
                attention_drop: float = 0,
                drop_rate: float = 0,
                bias: bool = True):
    """This is an example for args in moe self attention, since lots of modules should be
    adapted before putting them in experts.
    """
    dropout1 = Dropout(attention_drop, mode=ParallelMode.TENSOR)
    dropout2 = Dropout(drop_rate, mode=ParallelMode.TENSOR)
    return dict(
        d_model=d_model,
        n_heads=n_heads,
        d_kv=d_kv,
        bias=bias,
        dropout1=dropout1,
        dropout2=dropout2
    )


def moe_mlp_args(d_model: int,
                 d_ff: int,
                 drop_rate: float,
                 bias: bool = True):
    """This is an example for args of MLP in Experts, since lots of modules should be adapted
    before putting them in experts.
    """
    dropout1 = Dropout(drop_rate, mode=ParallelMode.TENSOR)
    dropout2 = Dropout(drop_rate, mode=ParallelMode.TENSOR)
    return dict(
        d_model=d_model,
        d_ff=d_ff,
        bias=bias,
        dropout1=dropout1,
        dropout2=dropout2
    )
