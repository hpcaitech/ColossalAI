from write_vqa import make_arrow
import os

root = os.path("/work/zhangyq/ColossalAI/examples/vilt/data/")
arrow_root = os.path.join(root, "arrow")
make_arrow(root, arrow_root)