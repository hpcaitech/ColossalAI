#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import inspect
from collections.abc import Iterable

from colossalai.registry import *



def build_from_config(module, config: dict):
    """Returns an object of :class:`module` constructed from `config`.

    Args:
        module: A python or user-defined class
        config: A python dict containing information used in the construction of the return object

    Returns: An ``object`` of interest

    Raises:
        AssertionError: Raises an AssertionError if `module` is not a class

    """
    assert inspect.isclass(module), 'module must be a class'
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
    assert isinstance(
        registry, Registry), f'Expected type Registry but got {type(registry)}'

    mod_type = config_.pop('type')
    assert registry.has(
        mod_type), f'{mod_type} is not found in registry {registry.name}'
    try:
        obj = registry.get_module(mod_type)(**config_)
    except Exception as e:
        print(
            f'An error occurred when building {mod_type} from registry {registry.name}',
            flush=True)
        raise e

    return obj



def build_layer(config):
    """Returns a layer object of :class:`nn.Module` constructed from `config`.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``LAYERS``.

    Returns:
        An object of :class:`torch.nn.Module`
    """
    return build_from_registry(config, LAYERS)


def build_loss(config):
    """Returns a loss function object of :class:`torch.autograd.Function` constructed
    from `config`.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``LOSSES``.

    Returns:
        An object of :class:`torch.nn.modules.loss._Loss`
    """
    return build_from_registry(config, LOSSES)


def build_model(config):
    """Returns a model object of :class:`nn.Module` constructed from `config`.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``MODELS``.

    Returns:
        An object of :class:`torch.nn.Module`
    """
    return build_from_registry(config, MODELS)


def build_dataset(config):
    """Returns a dataset object of :class:`torch.utils.data.Dataset` constructed
    from `config`.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``DATASETS``.

    Returns:
        An object of :class:`torch.utils.data.Dataset`
    """
    return build_from_registry(config, DATASETS)


def build_optimizer(config, model):
    """Returns an optimizer object of :class:`torch.optim.Optimizer` constructed from `config`,
    'model' and 'params'.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``OPTIMIZERS``.
        model (:class:`nn.Module`): A model containing parameters for the optimizer

    Returns:
        An object of :class:`torch.optim.Optimizer`
    """
    config_ = config.copy()
    config_['params'] = model.parameters()
    return build_from_registry(config_, OPTIMIZERS)


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
        An object of :class:`colossalai.engine.BaseGradientHandler`
    """
    config_ = config.copy()
    config_['model'] = model
    config_['optimizer'] = optimizer
    return build_from_registry(config_, GRADIENT_HANDLER)


def build_hooks(config, trainer):
    """Returns a hook object of :class:`BaseHook` constructed from `config` and `trainer`.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``HOOKS``.

    Returns:
        An object of :class:`colossalai.trainer.hooks.BaseHook`
    """
    config_ = config.copy()
    config_['trainer'] = trainer
    return build_from_registry(config_, HOOKS)


def build_ophooks(config):
    """Returns a hook object of :class:`BaseOpHook` constructed from `config`.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``OPHOOKS``.

    Returns:
        An object of :class:`colossalai.trainer.hooks.BaseOpHook`
    """
    config_ = config.copy()
    return build_from_registry(config_, OPHOOKS)


def build_transform(config):
    """Returns a transformation object of :class:`torchvision.transforms` constructed
    from `config`.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``TRANSFORMS``.

    Returns:
        An object of :class:`torchvision.transforms`
    """
    return build_from_registry(config, TRANSFORMS)


def build_data_sampler(config, dataset):
    """Returns a data sampler object of :class:`colossalai.nn.data.sampler.BaseSampler`
    constructed from `config`.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``DATA_SAMPLERS``.
        dataset (:class:`torch.utils.data.Dataset`): An object of
            :class:`torch.utils.data.Dataset` containing information
            used in the construction of the return object
    Returns:
        An object of :class:`colossalai.utils.data_sampler.BaseSampler`
    """
    config_ = config.copy()
    config_['dataset'] = dataset
    return build_from_registry(config_, DATA_SAMPLERS)


def build_lr_scheduler(config, optimizer):
    """Returns a learning rate scheduler object of :class:`torch.optim.lr_scheduler`
    constructed from `config`, `optimizer`, `total_steps` and `num_steps_per_epoch`.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``lr_schedule``.
        optimizer (:class:`torch.optim.Optimizer`): An optimizer object containing
            parameters for the learning rate scheduler.

    Returns:
        An object of :class:`torch.optim.lr_scheduler`
    """
    config_ = config.copy()
    config_['optimizer'] = optimizer
    return build_from_registry(config_, LR_SCHEDULERS)

def build_schedule(config):
    """Returns a schedule of :class:`colossalai.engine.schedule.BaseSchedule`.

    Args:
        config (dict or :class:`colossalai.context.Config`): A python dict or
            a :class:`colossalai.context.Config` object containing information
            used in the construction of the ``Schedule``.

    Returns:
        An object of :class:`colossalai.engine.schedule.BaseSchedule`
    """
    return build_from_registry(config, SCHEDULE)
