"""Verify the original cache path and support for a node-local symlink."""
import os
from pathlib import Path

from colossalai.kernel.extensions.base_extension import _Extension


def test_default_cache_path_and_local_symlink(tmp_path, monkeypatch):
    monkeypatch.delenv("COLOSSALAI_EXTENSION_CACHE", raising=False)
    original_path = Path(_Extension.get_jit_extension_folder_path())
    assert original_path.parent == Path(os.path.expanduser("~")) / ".cache/colossalai/torch_extensions"
    assert original_path.name.startswith("torch")
    monkeypatch.setattr(os.path, "expanduser", lambda _: str(tmp_path / "synthetic-home"))
    cache_path = Path(_Extension.get_jit_extension_folder_path())
    local_target = tmp_path / "node-local" / cache_path.name
    local_target.mkdir(parents=True)
    cache_path.parent.mkdir(parents=True)
    cache_path.symlink_to(local_target, target_is_directory=True)
    (cache_path / "probe").write_text("local")
    assert (local_target / "probe").read_text() == "local"


def test_extension_cache_env_override(tmp_path, monkeypatch):
    custom_root = tmp_path / "custom_ext_cache"
    monkeypatch.setenv("COLOSSALAI_EXTENSION_CACHE", str(custom_root))
    cache_path = Path(_Extension.get_jit_extension_folder_path())
    assert cache_path.parent == custom_root
    assert cache_path.name.startswith("torch")


def test_extension_cache_empty_or_whitespace_env_fallback(monkeypatch):
    for bad_val in ["", "   ", "\t\n"]:
        monkeypatch.setenv("COLOSSALAI_EXTENSION_CACHE", bad_val)
        cache_path = Path(_Extension.get_jit_extension_folder_path())
        assert cache_path.is_absolute()
        assert cache_path.parent == Path(os.path.expanduser("~")) / ".cache/colossalai/torch_extensions"
        assert cache_path.name.startswith("torch")


