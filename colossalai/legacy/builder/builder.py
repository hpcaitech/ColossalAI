#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import inspect

from colossalai.legacy.registry import *


def build_from_config(module, config: dict):
    """Returns an object of :class:`module` constructed from `config`.

    Args:
        module: A python or user-defined class
        config: A python dict containing information used in the construction of the return object

    Returns: An ``object`` of interest

    Raises:
        AssertionError: Raises an AssertionError if `module` is not a class

    """
    assert inspect.isclass(module), "module must be a class"
    return module(**config)


def build_from_registry(config, registry: Registry):
    r"""Returns an object constructed from `config`, the type of the object
    is specified by `registry`.

    Note:
        the `config` is used to construct the return object such as `LAYERS`, `OPTIMIZERS`
        and other support types in `registry`. The `config` should contain
        all required parameters of corresponding object. The details of support
        types in `registry` and the `mod_type` in `config` could be found in
        `registry <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/registry/__init__.py>`_.

    Args:
        config (dict or :class:`colossalai.context.colossalai.context.Config`): information
            used in the construction of the return object.
        registry (:class:`Registry`): A registry specifying the type of the return object

    Returns:
        A Python object specified by `registry`.

    Raises:
        Exception: Raises an Exception if an error occurred when building from registry.
    """
    config_ = config.copy()  # keep the original config untouched
    assert isinstance(registry, Registry), f"Expected type Registry but got {type(registry)}"

    mod_type = config_.pop("type")
    assert registry.has(mod_type), f"{mod_type} is not found in registry {registry.name}"
    try:
        obj = registry.get_module(mod_type)(**config_)
    except Exception as e:
        print(f"An error occurred when building {mod_type} from registry {registry.name}", flush=True)
        raise e

    return obj


def build_gradient_handler(config, model, optimizer):
    """Returns a gradient handler object of :class:`BaseGradientHandler` constructed from `config`,
    `model` and `optimizer`.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``GRADIENT_HANDLER``.
        model (:class:`nn.Module`): A model containing parameters for the gradient handler
        optimizer (:class:`torch.optim.Optimizer`): An optimizer object containing parameters for the gradient handler

    Returns:
        An object of :class:`colossalai.legacy.engine.BaseGradientHandler`
    """
    config_ = config.copy()
    config_["model"] = model
    config_["optimizer"] = optimizer
    return build_from_registry(config_, GRADIENT_HANDLER)
