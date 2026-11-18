"""Copy exact public Ouro source snapshots into small offline CI fixtures."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[3]
MODELS = {
    "ouro14": ("ByteDance/Ouro-1.4B", "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"),
    "ouro26": ("ByteDance/Ouro-2.6B", "1ed04250da1a9936042725d302e81c8fa2ab5abd"),
}
for key, (repo, revision) in MODELS.items():
    source = Path(snapshot_download(repo, revision=revision, local_files_only=True))
    target = ROOT / "tests/fixtures/ouro" / key
    target.mkdir(parents=True, exist_ok=True)
    files = [
        p
        for p in source.iterdir()
        if p.suffix == ".py" or p.name in ("config.json", "LICENSE", "LICENSE.txt", "README.md")
    ]
    manifest = {"repository": repo, "revision": revision, "license": "Apache-2.0", "sha256": {}}
    for path in sorted(files):
        data = path.read_bytes()
        assert len(data) < 1000000, path
        out = target / path.name
        if out.exists() and out.read_bytes() != data:
            raise RuntimeError(f"Refusing to replace an existing different fixture: {out}")
        out.write_bytes(data)
        manifest["sha256"][path.name] = hashlib.sha256(data).hexdigest()
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(key, revision, {p.name: p.stat().st_size for p in target.iterdir()})
