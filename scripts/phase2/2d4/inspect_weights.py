"""Check that all shards named in the pinned official weight indexes exist."""

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.kit.remote_models import REMOTE_MODELS

for key in ("ouro14", "ouro26"):
    repo, revision = REMOTE_MODELS[key]
    root = Path(snapshot_download(repo, revision=revision, local_files_only=True))
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    missing = [name for name in shards if not (root / name).is_file()]
    assert not missing, (key, missing)
    print(
        json.dumps(
            dict(model=repo, revision=revision, shards=shards, bytes=sum((root / n).stat().st_size for n in shards))
        )
    )
