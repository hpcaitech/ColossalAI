# Build PyTorch Extensions

## Overview

Building PyTorch extensions can be a difficult task for users not from the system background. It is definitely frustrating if the users encounter many strange technical jargons when install Colossal-AI. Therefore, we will provide two methods of building the PyTorch extensions for the users.

1. Build CUDA extensions when running `pip install` if `CUDA_EXT=1`
2. Build the extension during runtime

The first method is more suitable for users who are familiar with CUDA environment configurations. The second method is for those who are not as they only need to build the kernel which is required by their program.

These two methods have different advantages and disadvantages.
Method 1 is good because it allows the user to build all kernels during installation and directly import the kernel. They don't need to care about kernel building when running their program. However, installation may fail if they don't know how to configure their environments and this leads to much frustration.
Method 2 is good because it allows the user to only build the kernel they actually need, such that there is a lower probability that they encounter environment issue. However, it may slow down their program due to the first build and subsequence load.

## PyTorch Extensions in Colossal-AI

As mentioned in the section above, our aim is to make these two methods coherently supported in Colossal-AI, meaning that for a kernel should be either built in `setup.py` or during runtime.
There are mainly two functions used to build extensions.

1. `torch.utils.cpp_extension.CUDAExtension`: used to build extensions in `setup.py` during `pip install`.
2. `torch.utils.cpp_extension.load`: used to build and load extension during runtime

Please note that the extension build by `CUDAExtension` cannot be loaded by the `load` function and `load` will run its own build again (correct me if I am wrong).

We have implemented the following conventions:

1. All pre-built kernels (those installed with `setup.py`) will be found in `colossalai._C`
2. All runtime-built kernels will be found in the default torch extension path, i.e. ~/.cache/colossalai/torch_extensions. (If we put the built kernels in the installed site-package directory, this will make pip uninstall incomplete)

When loading the built kernel, we will first check if the pre-built one exists. If not, the runtime build will be triggered.
