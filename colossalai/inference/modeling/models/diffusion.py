import inspect
import types

import torch
from torch import nn


class DiffusionPipe(nn.Module):
    """
    This Class convert a class of `DiffusionPipeline` into `nn.Module` and reserve most of origin attr,function and property.
    """

    def __init__(self, source_obj) -> None:
        super(DiffusionPipe, self).__init__()

        for k, v in source_obj.__dict__.items():
            if isinstance(v, nn.Module):
                self.add_module(k, v)
            else:
                setattr(self, k, v)

        skip_list = ["_execution_device", "to", "device"]  # this

        for name, member in inspect.getmembers(source_obj.__class__):
            if name in skip_list:
                continue
            if not name.startswith("__") and not name.endswith("__"):
                if isinstance(member, property):
                    setattr(self.__class__, name, member)
                elif inspect.isfunction(member) or inspect.ismethod(member):
                    bound_method = types.MethodType(member, self)
                    setattr(self, name, bound_method)
                elif not callable(member) and not isinstance(member, property):
                    setattr(self, name, member)
            elif name == "__call__":
                bound_method = types.MethodType(member, self)
                setattr(self, "_forward", bound_method)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        # return self.device
        return torch.device("cuda")

    @property
    def device(self):
        next(self.parameters()).device

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    """
    below just for debug, remove later
    """

    def inspect(self, cls):
        """
        print the methods and attributes of one class
        """
        print("Attributes:")
        for name, value in inspect.getmembers(cls):
            if not name.startswith("__") and not name.endswith("__"):
                if not inspect.isroutine(value):
                    print(f"  {name}: {value}")

        print("\nMethods:")
        for name, value in inspect.getmembers(cls):
            if not name.startswith("__") and not name.endswith("__"):
                if inspect.isroutine(value):
                    if inspect.isfunction(value):
                        print(f"  {name} (function)")
                    elif inspect.ismethod(value):
                        print(f"  {name} (method)")


def compare_modules(module1: nn.Module, module2: nn.Module) -> bool:
    """
    Compare two nn.Module instances to check if their parameters and buffers are exactly the same.

    Args:
    - module1: The first nn.Module instance.
    - module2: The second nn.Module instance.

    Returns:
    - bool: True if all parameters and buffers are equal, False otherwise.
    """
    # First, check if the two modules have the same state dict keys
    params1 = module1.state_dict()
    params2 = module2.state_dict()

    if params1.keys() != params2.keys():
        print(f"params keys not same in {module1}")
        return False

    # Compare each parameter and buffer tensor
    for key in params1:
        if not torch.equal(params1[key], params2[key]):
            print(f"params {key} not equal")
            return False

    # Check if both modules have the same training mode
    if module1.training != module2.training:
        return False

    # Recursively compare submodules
    for submodule1, submodule2 in zip(module1.children(), module2.children()):
        if not compare_modules(submodule1, submodule2):
            return False

    return True


def compare_nn_modules(obj1, obj2) -> bool:
    """
    Compare all nn.Module instances in two objects to check if their parameters and buffers are exactly the same.

    Args:
    - obj1: The first object containing nn.Module instances.
    - obj2: The second object containing nn.Module instances.

    Returns:
    - bool: True if all nn.Module parameters and buffers are equal, False otherwise.
    """
    # modules1 = {name: module for name, module in obj1.__dict__.items() if isinstance(module, nn.Module)}
    modules1 = {name: module for name, module in obj1.named_children()}
    modules2 = {name: module for name, module in obj2.__dict__.items() if isinstance(module, nn.Module)}

    if modules1.keys() != modules2.keys():
        print(modules1.keys())
        print(modules2.keys())
        return False

    for name in modules1:
        if not compare_modules(modules1[name], modules2[name]):
            print(f"false: {name}")
            return False

    return True


if __name__ == "__main__":
    pass

    # pipe = StableDiffusion3Pipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
    # )
    # pipe = pipe.to("cuda")

    # pipe = DiffusionPipe(pipe)
    # print(pipe)
    # image = pipe(
    #     "A cat holding a sign that says hello world",
    #     # "A lady with a beautiful hat",
    #     negative_prompt="",
    #     # num_inference_steps=1,
    #     guidance_scale=7.0,
    #     # height=8096,
    #     # width=8096,
    #     # height=2048,
    #     # width=2048,
    # ).images[0]
    # image.save("cat.jpg")

    # prompt = "An astronaut riding a green horse"
    # pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")
    # pipe = DiffusionPipe(pipe)
    # images = pipe(prompt=prompt).images[0]
    # images.save("a.jpg")

    # setattr(pipe, "forward", types.MethodType(pixart_alpha_forward, pipe))
    # images = pipe(prompt=prompt)
    # images = pipe.forward(prompt=prompt)[0]
    # images.save("a.jpg")

    # prompt = "An astronaut riding a green horse"
    # pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
    # pipe1 = pipe.to("cuda")
    # pipe1 = DiffusionPipe(pipe1)

    # print("compare result: ", compare_nn_modules(pipe1, pipe.to("cuda")))
    # images = pipe1(prompt=prompt).images[0]
    # images.save("a.jpg")
