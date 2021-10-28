#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import inspect
from collections.abc import Iterable

from colossalai.registry import *


def build_from_config(module, config: dict):
    """Returns an object of :class:`module` constructed from `config`.

    :param module: A python or user-defined class
    :type module: class
    :param config: A python dict containing information used in the construction
        of the return object
    :type config: dict
    :raises AssertionError: Raises an AssertionError if `module` is not a class
    :return: An object of :class:`module`
    :rtype: :class:`module`
    """
    assert inspect.isclass(module), 'module must be a class'
    return module(**config)


def build_from_registry(config, registry: Registry):
    """Returns an object constructed from `config`, the type of the object
    is specified by `registry`.

    :param config: A python dict or a :class:`colossalai.context.Config` object 
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.colossalai.context.Config`
    :param registry: A registry specifying the type of the return object
    :type registry: :class:`Registry`
    :raises AssertionError: Raises an AssertionError if `registry` is not an object
        of :class:`Registry` or `mod_type` in `config` is not found in `registry`
    :raises Exception: Raises an Exception if an error occurred when building
        from registry
    :return: An object specified by `registry`
    :rtype: Python object specified by `registry`
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
            f'An error occurred when building {mod_type} from registry {registry.name}', flush=True)
        raise e

    return obj


def build_layer(config):
    """Returns a layer object of :class:`nn.Module` constructed from `config`.

    :param config: A python dict or a :class:`colossalai.context.Config` object
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :return: An object of :class:`nn.Module`
    :rtype: :class:`nn.Module`
    """
    return build_from_registry(config, LAYERS)


def build_loss(config):
    """Returns a loss function object of :class:`torch.autograd.Function` constructed 
    from `config`.

    :param config: A python dict or a :class:`colossalai.context.Config` object
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :return: An object of :class:`torch.autograd.Function`
    :rtype: :class:`torch.autograd.Function`
    """
    return build_from_registry(config, LOSSES)


def build_model(config):
    """Returns a model object of :class:`nn.Module` constructed from `config`.

    :param config: A python dict or a :class:`colossalai.context.Config` object
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :return: An object of :class:`nn.Module`
    :rtype: :class:`nn.Module`
    """
    return build_from_registry(config, MODELS)


def build_dataset(config):
    """Returns a dataset object of :class:`torch.utils.data.Dataset` constructed 
    from `config`.

    :param config: A python dict or a :class:`colossalai.context.Config` object
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :return: An object of :class:`torch.utils.data.Dataset`
    :rtype: :class:`torch.utils.data.Dataset`
    """
    return build_from_registry(config, DATASETS)


def build_optimizer(config, model, params: Iterable = None, need_module=False):
    """Returns an optimizer object of :class:`torch.optim.Optimizer` constructed from `config`, 
    'model' and 'params'.

    :param config: A python dict or a :class:`colossalai.context.Config` object 
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :param model: A model containing parameters for the optimizer 
    :type model: :class:`nn.Module`
    :param params: A dict containing parameters for the optimizer
    :type params: dict, optional
    :param need_module: Indicates whether the optimizer needs a module
    :type params: bool, optional
    :raises AssertionError: Raises an AssertionError if both `model` and `params` are None
    :return: An object of :class:`torch.optim.Optimizer`
    :rtype: :class:`torch.optim.Optimizer`
    """
    assert model is not None or params is not None, 'arguments model and params can not both be None'
    if need_module:
        config['module'] = model
    elif model is not None:
        config['params'] = model.parameters()
    elif params is not None:
        config['params'] = params

    return build_from_registry(config, OPTIMIZERS)


def build_gradient_handler(config, model, optimizer):
    """Returns a gradient handler object of :class:`BaseGradientHandler` constructed from `config`,
    `model` and `optimizer`.

    :param config: A python dict or a :class:`colossalai.context.Config` object
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :param model: A model containing parameters for the gradient handler
    :type model: :class:`nn.Module`
    :param optimizer: An optimizer object containing parameters for the gradient handler
    :type optimizer: :class:`torch.optim.Optimizer`
    :return: An object of :class:`BaseGradientHandler`
    :rtype: :class:`BaseGradientHandler`
    """
    config_ = config.copy()
    mod_type = config_.pop('type')
    return GRADIENT_HANDLER.get_module(mod_type)(model, optimizer, **config_)


def build_hooks(config, trainer):
    """Returns a hook object of :class:`BaseHook` constructed from `config` and `trainer`.

    :param config: A python dict or a :class:`colossalai.context.Config` object
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :param trainer: A :class:`Trainer` object containing parameters for the hook
    :type trainer: :class:`Trainer`
    :return: An object of :class:`BaseHook`
    :rtype: :class:`BaseHook`
    """
    config['trainer'] = trainer
    return build_from_registry(config, HOOKS)


def build_transform(config):
    """Returns a transformation object of :class:`torchvision.transforms` constructed
    from `config`.

    :param config: A python dict or a :class:`colossalai.context.Config` object
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :return: An object of :class:`torchvision.transforms`
    :rtype: :class:`torchvision.transforms`
    """
    return build_from_registry(config, TRANSFORMS)


def build_pipe_alloc_policy(config):
    """Returns a pipeline allocation policy object constructed from `config`.

    :param config: A python dict or a :class:`colossalai.context.Config` object
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :return: A pipeline allocation policy object
    :rtype: 
    """
    return build_from_registry(config, PIPE_ALLOC_POLICY)


def build_data_sampler(config, dataset):
    """Returns a data sampler object of :class:`colossalai.nn.data.sampler.BaseSampler`
    constructed from `config`.

    :param config: A python dict or a :class:`colossalai.context.Config` object
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :param dataset: An object of :class:`torch.utils.data.Dataset` containing information
        used in the construction of the return object
    :type dataset: :class:`torch.utils.data.Dataset`
    :return: An object of :class:`colossalai.nn.data.sampler.BaseSampler`
    :rtype: :class:`colossalai.nn.data.sampler.BaseSampler`
    """
    config_ = config.copy()
    mod_type = config_.pop('type')
    return SAMPLERS.get_module(mod_type)(dataset, **config_)


def build_optimizer_wrapper(config, optimizer, model=None):
    """Returns an optimizer wrapper object of :class:`torch.optim.Optimizer` constructed 
    from `config`, `model` and `optimizer`.

    :param config: A python dict or a :class:`colossalai.context.Config` object 
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :param optimizer: An optimizer object containing parameters for the gradient handler
    :type optimizer: :class:`torch.optim.Optimizer`
    :param model: A model containing parameters for the gradient handler
    :type model: :class:`nn.Module`, optional
    :return: An object of :class:`torch.optim.Optimizer`
    :rtype: :class:`torch.optim.Optimizer`
    """
    config_ = config.copy()
    mod_type = config_.pop('type')

    # LSG: special treatment for zeor level 3
    if mod_type == 'ZeroRedundancyOptimizer_Level_3':
        return OPTIMIZER_WRAPPERS.get_module(mod_type)(model, optimizer, **config_)
    else:
        return OPTIMIZER_WRAPPERS.get_module(mod_type)(optimizer, **config_)


def build_lr_scheduler(config, optimizer, total_steps, num_steps_per_epoch):
    """Returns a learning rate scheduler object of :class:`torch.optim.lr_scheduler` 
    constructed from `config`, `optimizer`, `total_steps` and `num_steps_per_epoch`.

    :param config: A python dict or a :class:`colossalai.context.Config` object 
        containing information used in the construction of the return object
    :type config: dict or :class:`colossalai.context.Config`
    :param optimizer: An optimizer object containing parameters for the learning rate
        scheduler
    :type optimizer: :class:`torch.optim.Optimizer`
    :param total_steps: Number of total steps of the learning rate scheduler
    :type total_steps: int
    :param num_steps_per_epoch: number of steps per epoch of the learning rate scheduler
    :type num_steps_per_epoch: int
    :return: An object of :class:`torch.optim.lr_scheduler`
    :rtype: :class:`torch.optim.lr_scheduler`
    """
    config_ = config.copy()
    mod_type = config_.pop('type')
    # warmup epochs will overwrite warmup steps
    if 'warmup_epochs' in config_:
        warmup_epochs = config_.pop('warmup_epochs')
        config_['warmup_steps'] = int(num_steps_per_epoch * warmup_epochs)
    return LR_SCHEDULERS.get_module(mod_type)(optimizer, total_steps, num_steps_per_epoch=num_steps_per_epoch,
                                              **config_)
