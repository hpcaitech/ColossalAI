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

        # self.inspect(source_obj)
        # self.inspect(self)

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

    # def print_class_members(self, cls):
    #     print(f"Class: {cls.__name__}\n")

    #     # 打印类的所有成员
    #     for name, member in inspect.getmembers(cls):
    #         if not name.startswith('__') and not name.endswith('__'):
    #             if isinstance(member, property):
    #                 print(f"Property: {name}")
    #             elif isinstance(member, types.FunctionType):
    #                 print(f"Method: {name}")
    #             elif isinstance(member, staticmethod):
    #                 print(f"Static Method: {name}")
    #             elif isinstance(member, classmethod):
    #                 print(f"Class Method: {name}")
    #             elif isinstance(member, types.BuiltinFunctionType):
    #                 print(f"Builtin Function: {name}")
    #             elif isinstance(member, Descriptor):
    #                 print(f"Descriptor: {name}")
    #             else:
    #                 print(f"Attribute: {name} = {member}")


if __name__ == "__main__":
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    pipe = DiffusionPipe(pipe)
    print(pipe)
    image = pipe(
        "A cat holding a sign that says hello world",
        # "A lady with a beautiful hat",
        negative_prompt="",
        # num_inference_steps=1,
        guidance_scale=7.0,
        # height=8096,
        # width=8096,
        # height=2048,
        # width=2048,
    ).images[0]
    image.save("cat.jpg")
