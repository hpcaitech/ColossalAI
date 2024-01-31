# 🚀 Accelerator

## 🔗 Table of Contents

- [🚀 Accelerator](#-accelerator)
  - [🔗 Table of Contents](#-table-of-contents)
  - [📚 Introduction](#-introduction)
  - [📌 Design and Acknowledgement](#-design-and-acknowledgement)

## 📚 Introduction

This module offers a layer of abstraction for ColossalAI. With this module, the user can easily switch between different accelerator backends, such as Nvidia GPUs, Huawei NPUs, etc. This module is an attempt to make users' code portable across different hardware platform with a simple `auto_set_accelerator()` API.

## 📌 Design and Acknowledgement

Our `accelerator` module is heavily inspired by [`deepspeed/accelerator`](https://www.deepspeed.ai/tutorials/accelerator-abstraction-interface/). We found that it is a very well-designed and well-structured module that can be easily integrated into our project. We would like to thank the DeepSpeed team for their great work.

We implemented this accelerator module from scratch. At the same time, we have implemented our own modifications:
1. we updated the accelerator API names to be aligned with PyTorch's native API names.
2. we did not include the `op builder` in the `accelerator`. Instead, we have reconstructed our `kernel` module to automatically match the accelerator and its corresponding kernel implementations, so as to make modules less tangled.
