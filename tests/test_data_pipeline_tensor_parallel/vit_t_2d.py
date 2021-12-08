
import sys
from pathlib import Path
repo_path = str(Path(__file__).absolute().parents[2])
sys.path.append(repo_path)

try:
    import model_zoo.vit.vision_transformer_from_config
except ImportError:
    raise ImportError("model_zoo is not found, please check your path")

IMG_SIZE = 32
PATCH_SIZE = 4
DIM = 512
NUM_ATTENTION_HEADS = 8
NUM_CLASSES = 10
DEPTH = 6

model_cfg = dict(
    type='VisionTransformerFromConfig',
    tensor_splitting_cfg=dict(
        type='ViTInputSplitter2D',
    ),
    embedding_cfg=dict(
        type='ViTPatchEmbedding2D',
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=DIM,
    ),
    token_fusion_cfg=dict(
        type='ViTTokenFuser2D',
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=DIM,
        drop_rate=0.1
    ),
    norm_cfg=dict(
        type='LayerNorm2D',
        normalized_shape=DIM,
        eps=1e-6,
    ),
    block_cfg=dict(
        type='ViTBlock',
        attention_cfg=dict(
            type='ViTSelfAttention2D',
            hidden_size=DIM,
            num_attention_heads=NUM_ATTENTION_HEADS,
            attention_dropout_prob=0.,
            hidden_dropout_prob=0.1,
        ),
        droppath_cfg=dict(
            type='VanillaViTDropPath',
        ),
        mlp_cfg=dict(
            type='ViTMLP2D',
            in_features=DIM,
            dropout_prob=0.1,
            mlp_ratio=1
        ),
        norm_cfg=dict(
            type='LayerNorm2D',
            normalized_shape=DIM,
            eps=1e-6,
        ),
    ),
    head_cfg=dict(
        type='ViTHead2D',
        hidden_size=DIM,
        num_classes=NUM_CLASSES,
    ),
    embed_dim=DIM,
    depth=DEPTH,
    drop_path_rate=0.,
)
