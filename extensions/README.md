# 🔌 Extensions

## 📌 Table of Contents

- [🔌 Extensions](#-extensions)
  - [📌 Table of Contents](#-table-of-contents)
  - [📚 Introduction](#-introduction)
  - [🪅 Design](#-design)
  - [🛠 API Usage](#-api-usage)
  - [🏗 Write a customized extension](#-write-a-customized-extension)
  - [✏️ Acknowledgement](#️-acknowledgement)

## 📚 Introduction

This module is a designed to offer extensions to the existing ColossalAI framework. It is designed to be a collection of high-performance kernels to speed up the training and inference process. Different from writing an individual kernel, the `extensions` module offers a layer of abstraction to collate kernels written in different compiler backends and for different hardware backends in an organized way. Please see the design and usage in the sections below.

## 🪅 Design

The `extensions` module is a sub-module of the `colossalai.kernel` module. This module is put at the project root directory so that it can be imported for AOT (ahead-of-time) build. At the same time, it is symbolically linked at the `colossalai.kernel.extensions` path for runtime build.

As we want to support multi-backend kernels, we have to consider multiple compiler options such as `torch.jit`, `CUDA`, `triton` and multiple hardware backends such as `CPU`, `GPU` and `NPU`. To make it easy for the users, we have abstract away the kernels into extensions and expose a single loader to the user for each kind of kernel.

For example, if the user wants to use the CPU Adam kernel, he can just call `load()` on the kernel loader. The kernel loader will automatically select the correct extension based on the current hardware and compiler backend. The user does not need to worry about the details of the kernel implementation. For example, if the user is using ARM CPU, then Arm kernel will be built and loaded. If it is a X86 CPU, then it is the X86 kernel that will be loaded.

```python
from colossalai.kernel.kernel_loader import CPUAdamLoader

# load the kernel compatible with the current hardware
kernel = CPUAdamLoader().load()
```

![](https://github.com/hpcaitech/public_assets/blob/main/colossalai/img/extensions.png?raw=true)

## 🛠 API Usage

To make the `colossalai.kernel` easy to use, we expose some simple APIs and you can use them based on your scenario.

- Case 1: Simply load a kernel

```python
from colossalai.kernel.kernel_loader import CPUAdamLoader

# load the kernel compatible with the current hardware
kernel = CPUAdamLoader().load()
```

- Case 2: Load a specific kernel

This case applies if you are familiar with the extensions available.

```python
from colossalai.kernel.kernel_loader import CPUAdamLoader

# load the kernel by giving the kernel name
kernel = CPUAdamLoader().load(ext_name="cpu_adam_arm")
```

- Case 3: Register your own extension

This case applies if you know how to write an extension. If you do not know how, you can refer to the section below.

```python
from colossalai.kernel.kernel_loader import CPUAdamLoader
from colossalai.kernel.base_extension import _Extension

# create your own extension class
class MyExtension(_Extension):

    def __init__(self):
        self._name = "my_extension"
        self._support_aot = True
        self._support_jit = True
        self.priority = 10

    # implementation here
    ...

# register your extension
# you can use the priority value to make sure your kernel will be loaded by default
CPUAdamLoader.register_extension(MyExtension)

# load the kernel
kernel = CPUAdamLoader().load()
```

## 🏗 Write a customized extension

It is easy to write a customized extension. If you have experience writing CUDA/triton kernels, you should get familiar with the process quickly.

You just need to inherit the `_Extension` base class or other backend-specific classes such as `_CudaExtension` and implement the abstract methods. Then, you need to register your extension to the kernel loader based on the Case 3 above. The kernel loader will automatically select the correct extension based on the priority score, current hardware, compiler backend.

```python
from colossalai.kernel.base_extension import _Extension


class MyExtension(_Extension):

    def __init__(self):
        self._name = "my_extension"
        self._support_aot = True
        self._support_jit = True
        self.priority = 10

    def is_available(self) -> bool:
        """
        Return if the required hardware can be found.
        """
        ...

    def assert_compatible(self) -> None:
        """
        Check if the hardware required by the kernel is compatible.
        """
        ...

    def build_aot(self) -> Union["CppExtension", "CUDAExtension"]:
        """
        If this kernel can be built AOT, it should return an extension object
        to Python setuptools for compilation.
        """
        ...

    def build_jit(self) -> Callable:
        """
        Build extension kernel just in time.
        """
        ...

    def load(self):
        """
        The API called by the user to get the kernel.
        """
        ...

```

## ✏️ Acknowledgement

This module is written from scratch but we learnt a lot by looking into [DeepSpeed'
s op_builder](https://github.com/microsoft/DeepSpeed/tree/master/op_builder). We wish to acknowledge their great work and contributions to the open-source community.
