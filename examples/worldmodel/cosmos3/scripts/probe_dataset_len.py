"""How many windows a given ``--max-episodes`` selection indexes.

Needed before pointing training at a latent cache: the sampler shuffles over the whole selection, so a cache that
covers fewer windows than this leaves gaps that only show up as a KeyError mid-run. Written as a file because
``dev-container-node.sh`` joins its arguments into one ``bash -c`` string and destroys inline quoting.
"""

import sys
from pathlib import Path

# Works whether this file sits inside the repository (four levels below the root) or is copied into a scripts
# directory outside it, where only PYTHONPATH can supply the package.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
except IndexError:
    pass

from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset

DATA = "/mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success"

for episodes in [int(value) for value in sys.argv[1:]] or [1, 4, 8]:
    dataset = DroidActionPolicyDataset(
        DATA, chunk_length=32, fps=15.0, resolution="480", augment=False, max_episodes=episodes
    )
    print(f"max_episodes={episodes:3d} windows={len(dataset)}", flush=True)
