"""
This code is copied from https://huggingface.co/THUDM/chatglm-6b/resolve/main/configuration_chatglm.py
"""

""" ChatGLM model configuration """

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ChatGLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~ChatGLMModel`].
    It is used to instantiate an ChatGLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the ChatGLM-6B [THUDM/ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 150528):
            Vocabulary size of the ChatGLM-6B model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~ChatGLMModel`] or
            [`~TFChatGLMModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        inner_hidden_size (`int`, *optional*, defaults to 16384):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        max_sequence_length (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        layernorm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models).
        Example:

    ```python
    >>> from configuration_chatglm import ChatGLMConfig
    >>> from modeling_chatglm import ChatGLMModel

    >>> # Initializing a ChatGLM-6B THUDM/ChatGLM-6B style configuration
    >>> configuration = ChatGLMConfig()

    >>> # Initializing a model from the THUDM/ChatGLM-6B style configuration
    >>> model = ChatGLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "chatglm"

    def __init__(
        self,
        vocab_size=130528,
        hidden_size=4096,
        num_layers=28,
        num_attention_heads=32,
        layernorm_epsilon=1e-5,
        use_cache=True,
        bos_token_id=130004,
        eos_token_id=130005,
        mask_token_id=130000,
        gmask_token_id=130001,
        pad_token_id=3,
        max_sequence_length=2048,
        inner_hidden_size=16384,
        position_encoding_2d=True,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.layernorm_epsilon = layernorm_epsilon
        self.inner_hidden_size = inner_hidden_size
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.gmask_token_id = gmask_token_id
        self.position_encoding_2d = position_encoding_2d
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection

        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
