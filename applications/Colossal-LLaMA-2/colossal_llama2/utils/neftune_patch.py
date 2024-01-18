#    Copyright 2023 The Hugging Face team
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch


def unwrap(model):
    return model.unwrap().module


def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for torch.nn.Embedding
    layers. This method is slightly adapted from the original source code that can be found here:
    https://github.com/neelsjain/NEFTune Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```
    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set `module.neftune_noise_alpha` to
            the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    if module.training:
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    return output


def activate_neftune(model, neftune_noise_alpha=0.1):
    r"""
    Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper:
    https://arxiv.org/abs/2310.05914
    """
    embeddings = unwrap(model).get_input_embeddings()

    embeddings.neftune_noise_alpha = neftune_noise_alpha
    hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
    neftune_hook_handle = hook_handle

    return model, neftune_hook_handle


def deactivate_neftune(model, neftune_hook_handle):
    """
    Deactivates the neftune method. Make sure to call `_activate_neftune` first.
    """
    embeddings = unwrap(model).get_input_embeddings()

    neftune_hook_handle.remove()
    del embeddings.neftune_noise_alpha
