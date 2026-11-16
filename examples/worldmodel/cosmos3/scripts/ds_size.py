"""Window count for a few --max-episodes values, to size a benchmark slice that does not wrap mid-screen."""

from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset

DATA = "/mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success"

for episodes in (64, 256, 1024):
    dataset = DroidActionPolicyDataset(DATA, chunk_length=32, fps=15.0, resolution="480", augment=False,
                                       max_episodes=episodes)
    per_rank = len(dataset) // 8
    print(f"max_episodes={episodes:5d} windows={len(dataset):6d} per_rank={per_rank:5d} "
          f"batches_of_32_per_rank={per_rank // 32}")
