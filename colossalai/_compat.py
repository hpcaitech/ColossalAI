r"""Internal transformers compatibility shim for ColossalAI (not exported).

Background
----------
ColossalAI pins transformers to ``==4.51.3`` (``requirements/requirements.txt``).
During the migration to v5, several referenced transformers internals changed in two ways:

1. **Moved** — for example, ``no_init_weights`` moved from ``transformers.modeling_utils``
   to ``transformers.initialization``.
2. **Removed** — ``is_remote_url`` / ``download_url`` / ``is_safetensors_available``
   have no remaining definitions in the v5 transformers installation tree
   (see ``scripts/N11_symbol_relocation_probe.py``).

This module centralizes both differences so version-specific branches do not spread across
``lazy`` / ``shardformer`` / ``checkpoint_io``.

Conventions (following existing repository style, not introducing a new pattern)
----------------------------------------------------------------------------------
* **Prefer feature detection over version strings**: moved symbols use two
  ``try/except ImportError`` import paths, covering v4 and v5 without parsing
  ``transformers.__version__``. An existing example is
  ``shardformer/policies/qwen2.py``, which uses
  ``hasattr(self.model.config, "num_key_value_heads")`` for feature detection.
* **Do not silently change semantics when a capability is removed**: return ``None`` and
  let the call site handle it explicitly and raise an explanatory error.
* **Importing this module has no side effects**: do not import heavyweight transformers
  modules or log at module scope. As with ``shardformer/_utils.py``, all ``get_*`` helpers
  resolve dependencies lazily inside functions.

Usage
-----
    from colossalai._compat import get_no_init_weights, is_safetensors_available

    with get_no_init_weights()():
        ...

    use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)
"""

from importlib import import_module
from importlib.util import find_spec
from inspect import signature
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

# Cache the result of `get_auth_kwarg_name` lazily to avoid inspecting the signature per load.
_AUTH_KWARG_NAME: Optional[str] = None
_IS_TRANSFORMERS_V5: Optional[bool] = None


def get_no_init_weights() -> Callable[[], Any]:
    r"""Return the ``no_init_weights`` context-manager factory.

    Transformers v5 defines ``no_init_weights`` in ``transformers.initialization``
    (v5.17.0: ``initialization.py:254``); v4 keeps it in ``transformers.modeling_utils``.

    Returns:
        Callable[[], Any]: A no-argument callable that returns a context manager.
    """
    try:
        from transformers.initialization import no_init_weights as _fn
    except ImportError:
        from transformers.modeling_utils import no_init_weights as _fn
    return _fn


def is_safetensors_available() -> bool:
    r"""Return whether ``safetensors`` is available.

    v4 provides ``transformers.utils.is_safetensors_available``; v5 removed that helper
    because safetensors became a hard dependency, so this falls back to ``importlib``.

    The repository also has a same-named implementation at
    ``colossalai/checkpoint_io/utils.py:62``. Its body is
    ``try: return True / except ImportError: return False`` and therefore always returns
    ``True`` (a remnant of an early module-level import). Do not reuse it here, or the
    availability check would become a constant.

    Returns:
        bool: ``True`` when the package is available.
    """
    try:
        from transformers.utils import is_safetensors_available as _fn
    except ImportError:
        return find_spec("safetensors") is not None
    return bool(_fn())


def is_remote_url(url_or_filename: str) -> bool:
    r"""Return whether a string is a raw URL (``http`` / ``https``).

    Transformers v5 removed this symbol, so it is implemented here with the v4 semantics
    (the v4 implementation also checks ``urlparse(...).scheme in ("http", "https")``).
    It is retained to support an explanatory error; see :func:`get_download_url`.

    Args:
        url_or_filename (str): Path or URL to inspect.

    Returns:
        bool: ``True`` when the input is a raw URL.
    """
    return urlparse(str(url_or_filename)).scheme in ("http", "https")


def get_download_url() -> Optional[Callable[..., str]]:
    r"""Return the implementation for downloading weights from a raw URL.

    v4 returns ``transformers.modeling_utils.download_url``. **v5 removed this
    capability**: ``download_url`` and ``is_remote_url`` have no remaining definitions,
    and ``transformers.utils.hub.cached_file`` now accepts
    ``(path_or_repo_id, filename, **kwargs)`` rather than URLs.

    Therefore this returns ``None`` under v5. Callers must handle ``None`` explicitly and
    raise an explanatory error instead of silently treating the URL as a repository ID,
    which would expose an opaque ``OSError``.

    Returns:
        Optional[Callable[..., str]]: The v4 callable, or ``None`` under v5.
    """
    try:
        from transformers.modeling_utils import download_url as _fn
    except ImportError:
        return None
    return _fn


def get_auth_kwarg_name() -> str:
    r"""Return the authentication parameter name used by the installed transformers.

    v5 renamed ``use_auth_token`` to ``token``. On v5.17.0, all four relevant entry points
    expose ``token`` and no longer expose ``use_auth_token``:

    * ``PretrainedConfig.from_pretrained``
    * ``GenerationConfig.from_pretrained``
    * ``transformers.utils.hub.has_file``
    * ``transformers.utils.hub.cached_file``

    ``cached_file`` itself accepts ``(path_or_repo_id, filename, **kwargs)`` and forwards
    kwargs to ``cached_files``, which accepts ``token``.

    This is required because ``lazy/pretrained.py`` passes the argument explicitly to
    ``PretrainedConfig.from_pretrained(..., return_unused_kwargs=True)``. v4 recognizes it
    and does not return it as unused; v5 would return it in ``model_kwargs`` as an unknown
    argument and eventually raise ``TypeError: ... unexpected keyword argument
    'use_auth_token'``. **Even omitting the argument triggers the failure**, because the
    returned value is ``None``.

    Probe ``token`` first. In transitional versions that expose both names, this selects the
    new name without triggering a deprecation warning for ``use_auth_token``.

    Returns:
        str: ``"token"`` or ``"use_auth_token"``. Falls back to ``"token"`` if signature
            inspection fails.
    """
    global _AUTH_KWARG_NAME
    if _AUTH_KWARG_NAME is None:
        try:
            from transformers.configuration_utils import PretrainedConfig as _Config

            _params = signature(_Config.from_pretrained).parameters
            _AUTH_KWARG_NAME = "token" if "token" in _params else "use_auth_token"
        except Exception:
            _AUTH_KWARG_NAME = "token"
    return _AUTH_KWARG_NAME


def auth_kwargs(token: Any = None) -> Dict[str, Any]:
    r"""Build authentication kwargs with the name used by the installed transformers.

    Call sites replace ``use_auth_token=use_auth_token`` with::

        config, model_kwargs = cls.config_class.from_pretrained(
            config_path, ..., **auth_kwargs(use_auth_token)
        )

    Args:
        token (Any): Authentication token. ``None`` means unauthenticated, but is still
            passed explicitly; see :func:`get_auth_kwarg_name`.

    Returns:
        Dict[str, Any]: Either ``{"token": token}`` or ``{"use_auth_token": token}``.
    """
    return {get_auth_kwarg_name(): token}


def is_transformers_v5() -> bool:
    r"""Return whether the installed transformers is v5 or later.

    This is a deliberate exception to the feature-detection convention above. It picks a whole
    implementation (see :func:`reexport_transformers_impl`), and the boundary between the two is
    the v5 architecture change rather than any single symbol: probing e.g.
    ``transformers.masking_utils`` would route late v4 releases, which already ship it, to the v5
    implementation.

    Returns:
        bool: ``True`` when ``transformers.__version__`` has major version 5 or later.
    """
    global _IS_TRANSFORMERS_V5
    if _IS_TRANSFORMERS_V5 is None:
        import transformers

        _IS_TRANSFORMERS_V5 = int(transformers.__version__.split(".", 1)[0]) >= 5
    return _IS_TRANSFORMERS_V5


def reexport_transformers_impl(namespace: Dict[str, Any], package: str, stem: str) -> None:
    r"""Fill a module namespace with the implementation matching the transformers major version.

    Model families whose v4 and v5 implementations diverge keep them in two sibling modules,
    ``<stem>_tf4`` (the pinned v4, ``transformers==4.51.3``) and ``<stem>_tf5``. The public
    module calls this helper so that existing imports of it keep working under both versions.
    Every name except dunders is copied, private helpers included, because tests and policies
    import some of them.

    The names are references to the implementation module's objects, and its functions resolve
    globals in that module. Patching (``mock.patch``, ``monkeypatch.setattr``) must therefore
    target ``<stem>_tf4`` / ``<stem>_tf5``; patching the public module does not change behaviour.

    Args:
        namespace (Dict[str, Any]): ``globals()`` of the public module.
        package (str): ``__package__`` of the public module.
        stem (str): Module name prefix, e.g. ``"_llama"``.
    """
    impl = import_module(f"{package}.{stem}_tf5" if is_transformers_v5() else f"{package}.{stem}_tf4")
    namespace.update({k: v for k, v in vars(impl).items() if not (k.startswith("__") and k.endswith("__"))})
