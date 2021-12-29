
import sys
from pathlib import Path
repo_path = Path(__file__).absolute().parents[2]
sys.path.append(str(repo_path))

try:
    import model_zoo.vit.vision_transformer_from_config
except ImportError:
    raise ImportError("model_zoo is not found, please check your path")

BATCH_SIZE = 8
IMG_SIZE = 32
PATCH_SIZE = 4
DIM = 512
NUM_ATTENTION_HEADS = 8
SUMMA_DIM = 2
NUM_CLASSES = 10
DEPTH = 6
