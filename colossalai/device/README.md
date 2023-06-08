# 🗄 Device

## 📚 Table of Contents

- [🗄 Device](#-device)
  - [📚 Table of Contents](#-table-of-contents)
  - [🔗 Introduction](#-introduction)
  - [📝 Design](#-design)
  - [🔨 Usage](#-usage)

## 🔗 Introduction

This module contains the implementation of the abstraction of the device topology. It is used to represent the device topology and manage the distributed information related to the network.

## 📝 Design


This module is inspired by the DeviceMesh in the [Alpa project](https://github.com/alpa-projects/alpa) and the device array can be represented as a 1D or 2D mesh. We will be extending the device mesh to support 3D mesh in the future.


## 🔨 Usage

- Create a device mesh

```python
# this is the list of global ranks involved in the device mesh
# assume we have 4 GPUs and the global ranks for these GPUs are 0, 1, 2, 3
physical_mesh_id = torch.arange(4)
mesh_shape = [2, 2]
device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
```

- View the mesh


```python
# view the mesh shape
# expect output
# [2, 2]
print(device_mesh.shape)


# view the logical mesh with global ranks
# expect output
# [
#   [0, 1],
#   [2, 3]
# ]
print(device_mesh.logical_mesh_id)

# view the number of devices in the mesh
# expect output
# 4
print(device_mesh.num_devices)

```

- Initialize the process group

```python
# intialize process group
device_mesh.init_logical_process_group()


# get the process group for a rank with respect to an axis
# this is the process group involving global ranks 0 and 2
print(device_mesh.get_process_group(axis=0, global_rank=0))

# get the ranks in the process with respect to an axis
# expect output
# [0, 2]
print(device_mesh.get_ranks_in_process_group(axis=0, global_rank=0))
```
