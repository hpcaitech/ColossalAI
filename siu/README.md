# Siu: Solver integration utils for Colossal-AI

# Overview
This repo is somehow the minimal implementation of Colossal-AI FX. Features include:
- symbolic_trace()
  - Robust Control-flow Tracing / Recompile
  - Robust Activation Checkpoint Tracing / CodeGen
  - Easy-to-define Bias-Addition Split
- symbolic_profile()
  - Support ``MetaTensorMode``, where all Tensor operations are executed symbolically.
  - Shape Inference Across Device and Unified ``MetaInfo``
  - Ideal Flop Counter https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505

# Install
```bash
git clone https://github.com/super-dainiu/siu.git
cd siu
pip install -r requirements.txt
pip install -e .
```

# Quickstart
## siu.FX
**Reference:**

  https://pytorch.org/docs/stable/fx.html [[paper](https://arxiv.org/pdf/2112.08429)]


torch.FX is a toolkit for developers to use to transform nn.Module instances. FX consists of three main components: a symbolic tracer, an intermediate representation, and Python code generation. FX.Tracer hacks _\_\_torch_function\_\__ and use a Proxy object to propagate through any forward function of torch.nn.Module.
![image](https://user-images.githubusercontent.com/78588128/212531495-bbb934dd-dbbb-4578-8869-6171973f7dd8.png)
ColossalAI FX is modified from torch.FX, with the extra capability of ahead-of-time profiling enabled by the subclass of ``MetaTensor``.

### siu.fx.symbolic_trace()
A drawback of the original torch.FX implementation is that it is poor at handling control flow. All control flow is not PyTorch native operands and requires actual instances that specify the branches to execute on. For example,

```python
class MyModule(nn.Module):
    def forward(self, x):
        if x.dim() == 3:
            return x * 2 + 1
        else:
            return x - 5
```

The above function has the computation graph of

![image](https://user-images.githubusercontent.com/78588128/212532631-dba30734-577b-4418-8dc9-004d7983abc5.png)

However, since Proxy does not have concrete data, applying ``x.dim()`` will return nothing. In the context of the auto-parallel system, at least the control-flow dependencies for tensor shape should be removed, since any searched strategy could only auto-parallelize a specific computation graph with the same tensor shape. It is native to attach concrete data onto a Proxy, and propagate them through control flow.

![image](https://user-images.githubusercontent.com/78588128/212533403-1b620986-1c3a-420a-87c6-d08c9702135d.png)


With ``MetaTensor``, the computation during shape propagation can be virtualized. This speeds up tracing by avoiding allocating actual memory on devices.

#### Remarks
There is no free lunch for PyTorch to unify all operands in both its repo and other repos in its eco-system. For example, the einops library currently has no intention to support torch.FX (See https://github.com/arogozhnikov/einops/issues/188). To support different PyTorch-based libraries without modifying source code, good practices can be to allow users to register their implementation to substitute the functions not supported by torch.FX, or to avoid entering incompatible submodules.

### siu.fx.symbolic_profile()

``symbolic_profile`` is another important feature of Colossal-AI's auto-parallel system. Profiling DNN can be costly, as you need to allocate memory and execute on real devices. However, since the profiling requirements for auto-parallel is enough if we can detect when and where the intermediate activations (i.e. Tensor) are generated, we can profile the whole procedure without actually executing it. ``symbolic_profile``, as its name infers, profiles the whole network with symbolic information only.

```python
with MetaTensorMode():
    model = MyModule().cuda()
    sample = torch.rand(100, 3, 224, 224).cuda()
meta_args = dict(
    x = sample,
)
gm = symbolic_trace(model, meta_args=meta_args)
gm = symbolic_profile(gm, sample)
```

``symbolic_profile`` is enabled by ``ShapeProp`` and ``GraphProfile``.

#### ShapeProp
Both Tensor Parallel and Activation Checkpoint solvers need to know the shape information ahead of time. Unlike PyTorch's implementation, this ``ShapeProp`` can be executed under MetaTensorMode. With this, all the preparation for auto-parallel solvers can be done in milliseconds.

If a ``Graph`` is modified with some non-PyTorch functions, such as fused operands, you can register the shape propagation rule with the decorator.

```python
@register_shape_impl(fuse_conv_bn)
def fuse_conv_bn_shape_impl(*args, **kwargs):
     # do something here
     return torch.empty(output_shape, device=output_device)
```

An important notice is that ``ShapeProp`` will attach additional information to the graph, which will be exactly the input of ``GraphProfile``.

#### GraphProfile
``GraphProfile`` executes at the node level, and profiles both forward and backward within one node. However, the drawbacks of FX is that not every ``call_function`` saves its input for backward, and different tensor that flows within one FX.Graph can actually have the same layout. This raises problems for fine-grained profiling.

![image](https://user-images.githubusercontent.com/78588128/215312957-7eb6cbc3-61b2-49cf-95a4-6b859149eb8d.png)

To address this problem, I came up with a simulated environment enabled by ``torch.autograd.graph.saved_tensor_hooks`` and fake ``data_ptr``.
```python
class sim_env(saved_tensors_hooks):

    def __init__(self):
        super().__init__(self.pack_hook, self.unpack_hook)
        self.ctx = {}

    def pack_hook(self, tensor: torch.Tensor):
        self.ctx[tensor.data_ptr()] = tensor._tensor if hasattr(tensor, '_tensor') else tensor
        return tensor

    def unpack_hook(self, tensor):
        return tensor
```
The ``ctx`` variable will keep track of all saved tensors with a unique identifier. For non-profileable nodes, you can add your customized profile rules to it.

```python
@register_profile_impl(fuse_conv_bn)
def fuse_conv_bn_profile_impl(*args, **kwargs):
     # do something here
     return fwd_flop, bwd_flop, fwd_comm, bwd_comm
```

To provide a general insight of the profiled results, you can set ``verbose=True`` to print the summary as well.
```python
model = tm.resnet18()
sample = torch.rand(100, 3, 224, 224)
meta_args = dict(x=sample)
gm = symbolic_trace(model, meta_args=meta_args)
gm = symbolic_profile(gm, sample, verbose=True)

============================================================ Results =====================================================================
      Op type                    Op    Total size    Output size    Temp size    Param size    Backward size      Fwd FLOPs      Bwd FLOPs
-------------  --------------------  ------------  -------------  -----------  ------------  ---------------  -------------  -------------
  placeholder                     x      57.42 Mb       57.42 Mb          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module          features_0_0     153.13 Mb      153.12 Mb          0 b       3.38 Kb         57.42 Mb    1.08 GFLOPs    1.08 GFLOPs
  call_module          features_0_1     153.13 Mb      153.12 Mb          0 b         256 b        153.12 Mb  200.70 MFLOPs  200.70 MFLOPs
  call_module          features_0_2     153.12 Mb      153.12 Mb          0 b           0 b        153.12 Mb   40.14 MFLOPs        0 FLOPs
  call_module   features_1_conv_0_0     153.13 Mb      153.12 Mb          0 b       1.12 Kb        153.12 Mb  361.27 MFLOPs   11.56 GFLOPs
  call_module   features_1_conv_0_1     153.13 Mb      153.12 Mb          0 b         256 b        153.12 Mb  200.70 MFLOPs  200.70 MFLOPs
  call_module   features_1_conv_0_2     153.12 Mb      153.12 Mb          0 b           0 b        153.12 Mb   40.14 MFLOPs        0 FLOPs
  call_module     features_1_conv_1      76.56 Mb       76.56 Mb          0 b       2.00 Kb        153.12 Mb  642.25 MFLOPs  642.25 MFLOPs
  call_module     features_1_conv_2      76.56 Mb       76.56 Mb          0 b         128 b         76.56 Mb  100.35 MFLOPs  100.35 MFLOPs
  call_module   features_2_conv_0_0     459.38 Mb      459.38 Mb          0 b       6.00 Kb         76.56 Mb    1.93 GFLOPs    1.93 GFLOPs
  call_module   features_2_conv_0_1     459.38 Mb      459.38 Mb          0 b         768 b        459.38 Mb  602.11 MFLOPs  602.11 MFLOPs
  call_module   features_2_conv_0_2     459.38 Mb      459.38 Mb          0 b           0 b        459.38 Mb  120.42 MFLOPs        0 FLOPs
  call_module   features_2_conv_1_0     114.85 Mb      114.84 Mb          0 b       3.38 Kb        459.38 Mb  270.95 MFLOPs   26.01 GFLOPs
  call_module   features_2_conv_1_1     114.85 Mb      114.84 Mb          0 b         768 b        114.84 Mb  150.53 MFLOPs  150.53 MFLOPs
  call_module   features_2_conv_1_2     114.84 Mb      114.84 Mb          0 b           0 b        114.84 Mb   30.11 MFLOPs        0 FLOPs
  call_module     features_2_conv_2      28.72 Mb       28.71 Mb          0 b       9.00 Kb        114.84 Mb  722.53 MFLOPs  722.53 MFLOPs
  call_module     features_2_conv_3      28.71 Mb       28.71 Mb          0 b         192 b         28.71 Mb   37.63 MFLOPs   37.63 MFLOPs
  call_module   features_3_conv_0_0     172.28 Mb      172.27 Mb          0 b      13.50 Kb         28.71 Mb    1.08 GFLOPs    1.08 GFLOPs
  call_module   features_3_conv_0_1     172.27 Mb      172.27 Mb          0 b       1.12 Kb        172.27 Mb  225.79 MFLOPs  225.79 MFLOPs
  call_module   features_3_conv_0_2     172.27 Mb      172.27 Mb          0 b           0 b        172.27 Mb   45.16 MFLOPs        0 FLOPs
  call_module   features_3_conv_1_0     172.27 Mb      172.27 Mb          0 b       5.06 Kb        172.27 Mb  406.43 MFLOPs   58.53 GFLOPs
  call_module   features_3_conv_1_1     172.27 Mb      172.27 Mb          0 b       1.12 Kb        172.27 Mb  225.79 MFLOPs  225.79 MFLOPs
  call_module   features_3_conv_1_2     172.27 Mb      172.27 Mb          0 b           0 b        172.27 Mb   45.16 MFLOPs        0 FLOPs
  call_module     features_3_conv_2      28.72 Mb       28.71 Mb          0 b      13.50 Kb        172.27 Mb    1.08 GFLOPs    1.08 GFLOPs
  call_module     features_3_conv_3         480 b            0 b     28.71 Mb         192 b         28.71 Mb   37.63 MFLOPs   37.63 MFLOPs
call_function                   add      28.71 Mb       28.71 Mb          0 b           0 b         57.42 Mb    7.53 MFLOPs        0 FLOPs
  call_module   features_4_conv_0_0     172.28 Mb      172.27 Mb          0 b      13.50 Kb         28.71 Mb    1.08 GFLOPs    1.08 GFLOPs
  call_module   features_4_conv_0_1     172.27 Mb      172.27 Mb          0 b       1.12 Kb        172.27 Mb  225.79 MFLOPs  225.79 MFLOPs
  call_module   features_4_conv_0_2     172.27 Mb      172.27 Mb          0 b           0 b        172.27 Mb   45.16 MFLOPs        0 FLOPs
  call_module   features_4_conv_1_0      43.07 Mb       43.07 Mb          0 b       5.06 Kb        172.27 Mb  101.61 MFLOPs   14.63 GFLOPs
  call_module   features_4_conv_1_1      43.07 Mb       43.07 Mb          0 b       1.12 Kb         43.07 Mb   56.45 MFLOPs   56.45 MFLOPs
  call_module   features_4_conv_1_2      43.07 Mb       43.07 Mb          0 b           0 b         43.07 Mb   11.29 MFLOPs        0 FLOPs
  call_module     features_4_conv_2       9.59 Mb        9.57 Mb          0 b      18.00 Kb         43.07 Mb  361.27 MFLOPs  361.27 MFLOPs
  call_module     features_4_conv_3       9.57 Mb        9.57 Mb          0 b         256 b          9.57 Mb   12.54 MFLOPs   12.54 MFLOPs
  call_module   features_5_conv_0_0      57.45 Mb       57.42 Mb          0 b      24.00 Kb          9.57 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module   features_5_conv_0_1      57.43 Mb       57.42 Mb          0 b       1.50 Kb         57.42 Mb   75.26 MFLOPs   75.26 MFLOPs
  call_module   features_5_conv_0_2      57.42 Mb       57.42 Mb          0 b           0 b         57.42 Mb   15.05 MFLOPs        0 FLOPs
  call_module   features_5_conv_1_0      57.43 Mb       57.42 Mb          0 b       6.75 Kb         57.42 Mb  135.48 MFLOPs   26.01 GFLOPs
  call_module   features_5_conv_1_1      57.43 Mb       57.42 Mb          0 b       1.50 Kb         57.42 Mb   75.26 MFLOPs   75.26 MFLOPs
  call_module   features_5_conv_1_2      57.42 Mb       57.42 Mb          0 b           0 b         57.42 Mb   15.05 MFLOPs        0 FLOPs
  call_module     features_5_conv_2       9.59 Mb        9.57 Mb          0 b      24.00 Kb         57.42 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module     features_5_conv_3         640 b            0 b      9.57 Mb         256 b          9.57 Mb   12.54 MFLOPs   12.54 MFLOPs
call_function                 add_1       9.57 Mb        9.57 Mb          0 b           0 b         19.14 Mb    2.51 MFLOPs        0 FLOPs
  call_module   features_6_conv_0_0      57.45 Mb       57.42 Mb          0 b      24.00 Kb          9.57 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module   features_6_conv_0_1      57.43 Mb       57.42 Mb          0 b       1.50 Kb         57.42 Mb   75.26 MFLOPs   75.26 MFLOPs
  call_module   features_6_conv_0_2      57.42 Mb       57.42 Mb          0 b           0 b         57.42 Mb   15.05 MFLOPs        0 FLOPs
  call_module   features_6_conv_1_0      57.43 Mb       57.42 Mb          0 b       6.75 Kb         57.42 Mb  135.48 MFLOPs   26.01 GFLOPs
  call_module   features_6_conv_1_1      57.43 Mb       57.42 Mb          0 b       1.50 Kb         57.42 Mb   75.26 MFLOPs   75.26 MFLOPs
  call_module   features_6_conv_1_2      57.42 Mb       57.42 Mb          0 b           0 b         57.42 Mb   15.05 MFLOPs        0 FLOPs
  call_module     features_6_conv_2       9.59 Mb        9.57 Mb          0 b      24.00 Kb         57.42 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module     features_6_conv_3         640 b            0 b      9.57 Mb         256 b          9.57 Mb   12.54 MFLOPs   12.54 MFLOPs
call_function                 add_2       9.57 Mb        9.57 Mb          0 b           0 b         19.14 Mb    2.51 MFLOPs        0 FLOPs
  call_module   features_7_conv_0_0      57.45 Mb       57.42 Mb          0 b      24.00 Kb          9.57 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module   features_7_conv_0_1      57.43 Mb       57.42 Mb          0 b       1.50 Kb         57.42 Mb   75.26 MFLOPs   75.26 MFLOPs
  call_module   features_7_conv_0_2      57.42 Mb       57.42 Mb          0 b           0 b         57.42 Mb   15.05 MFLOPs        0 FLOPs
  call_module   features_7_conv_1_0      14.36 Mb       14.36 Mb          0 b       6.75 Kb         57.42 Mb   33.87 MFLOPs    6.50 GFLOPs
  call_module   features_7_conv_1_1      14.36 Mb       14.36 Mb          0 b       1.50 Kb         14.36 Mb   18.82 MFLOPs   18.82 MFLOPs
  call_module   features_7_conv_1_2      14.36 Mb       14.36 Mb          0 b           0 b         14.36 Mb    3.76 MFLOPs        0 FLOPs
  call_module     features_7_conv_2       4.83 Mb        4.79 Mb          0 b      48.00 Kb         14.36 Mb  240.84 MFLOPs  240.84 MFLOPs
  call_module     features_7_conv_3       4.79 Mb        4.79 Mb          0 b         512 b          4.79 Mb    6.27 MFLOPs    6.27 MFLOPs
  call_module   features_8_conv_0_0      28.80 Mb       28.71 Mb          0 b      96.00 Kb          4.79 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module   features_8_conv_0_1      28.72 Mb       28.71 Mb          0 b       3.00 Kb         28.71 Mb   37.63 MFLOPs   37.63 MFLOPs
  call_module   features_8_conv_0_2      28.71 Mb       28.71 Mb          0 b           0 b         28.71 Mb    7.53 MFLOPs        0 FLOPs
  call_module   features_8_conv_1_0      28.72 Mb       28.71 Mb          0 b      13.50 Kb         28.71 Mb   67.74 MFLOPs   26.01 GFLOPs
  call_module   features_8_conv_1_1      28.72 Mb       28.71 Mb          0 b       3.00 Kb         28.71 Mb   37.63 MFLOPs   37.63 MFLOPs
  call_module   features_8_conv_1_2      28.71 Mb       28.71 Mb          0 b           0 b         28.71 Mb    7.53 MFLOPs        0 FLOPs
  call_module     features_8_conv_2       4.88 Mb        4.79 Mb          0 b      96.00 Kb         28.71 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module     features_8_conv_3       1.25 Kb            0 b      4.79 Mb         512 b          4.79 Mb    6.27 MFLOPs    6.27 MFLOPs
call_function                 add_3       4.79 Mb        4.79 Mb          0 b           0 b          9.57 Mb    1.25 MFLOPs        0 FLOPs
  call_module   features_9_conv_0_0      28.80 Mb       28.71 Mb          0 b      96.00 Kb          4.79 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module   features_9_conv_0_1      28.72 Mb       28.71 Mb          0 b       3.00 Kb         28.71 Mb   37.63 MFLOPs   37.63 MFLOPs
  call_module   features_9_conv_0_2      28.71 Mb       28.71 Mb          0 b           0 b         28.71 Mb    7.53 MFLOPs        0 FLOPs
  call_module   features_9_conv_1_0      28.72 Mb       28.71 Mb          0 b      13.50 Kb         28.71 Mb   67.74 MFLOPs   26.01 GFLOPs
  call_module   features_9_conv_1_1      28.72 Mb       28.71 Mb          0 b       3.00 Kb         28.71 Mb   37.63 MFLOPs   37.63 MFLOPs
  call_module   features_9_conv_1_2      28.71 Mb       28.71 Mb          0 b           0 b         28.71 Mb    7.53 MFLOPs        0 FLOPs
  call_module     features_9_conv_2       4.88 Mb        4.79 Mb          0 b      96.00 Kb         28.71 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module     features_9_conv_3       1.25 Kb            0 b      4.79 Mb         512 b          4.79 Mb    6.27 MFLOPs    6.27 MFLOPs
call_function                 add_4       4.79 Mb        4.79 Mb          0 b           0 b          9.57 Mb    1.25 MFLOPs        0 FLOPs
  call_module  features_10_conv_0_0      28.80 Mb       28.71 Mb          0 b      96.00 Kb          4.79 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module  features_10_conv_0_1      28.72 Mb       28.71 Mb          0 b       3.00 Kb         28.71 Mb   37.63 MFLOPs   37.63 MFLOPs
  call_module  features_10_conv_0_2      28.71 Mb       28.71 Mb          0 b           0 b         28.71 Mb    7.53 MFLOPs        0 FLOPs
  call_module  features_10_conv_1_0      28.72 Mb       28.71 Mb          0 b      13.50 Kb         28.71 Mb   67.74 MFLOPs   26.01 GFLOPs
  call_module  features_10_conv_1_1      28.72 Mb       28.71 Mb          0 b       3.00 Kb         28.71 Mb   37.63 MFLOPs   37.63 MFLOPs
  call_module  features_10_conv_1_2      28.71 Mb       28.71 Mb          0 b           0 b         28.71 Mb    7.53 MFLOPs        0 FLOPs
  call_module    features_10_conv_2       4.88 Mb        4.79 Mb          0 b      96.00 Kb         28.71 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module    features_10_conv_3       1.25 Kb            0 b      4.79 Mb         512 b          4.79 Mb    6.27 MFLOPs    6.27 MFLOPs
call_function                 add_5       4.79 Mb        4.79 Mb          0 b           0 b          9.57 Mb    1.25 MFLOPs        0 FLOPs
  call_module  features_11_conv_0_0      28.80 Mb       28.71 Mb          0 b      96.00 Kb          4.79 Mb  481.69 MFLOPs  481.69 MFLOPs
  call_module  features_11_conv_0_1      28.72 Mb       28.71 Mb          0 b       3.00 Kb         28.71 Mb   37.63 MFLOPs   37.63 MFLOPs
  call_module  features_11_conv_0_2      28.71 Mb       28.71 Mb          0 b           0 b         28.71 Mb    7.53 MFLOPs        0 FLOPs
  call_module  features_11_conv_1_0      28.72 Mb       28.71 Mb          0 b      13.50 Kb         28.71 Mb   67.74 MFLOPs   26.01 GFLOPs
  call_module  features_11_conv_1_1      28.72 Mb       28.71 Mb          0 b       3.00 Kb         28.71 Mb   37.63 MFLOPs   37.63 MFLOPs
  call_module  features_11_conv_1_2      28.71 Mb       28.71 Mb          0 b           0 b         28.71 Mb    7.53 MFLOPs        0 FLOPs
  call_module    features_11_conv_2       7.32 Mb        7.18 Mb          0 b     144.00 Kb         28.71 Mb  722.53 MFLOPs  722.53 MFLOPs
  call_module    features_11_conv_3       7.18 Mb        7.18 Mb          0 b         768 b          7.18 Mb    9.41 MFLOPs    9.41 MFLOPs
  call_module  features_12_conv_0_0      43.28 Mb       43.07 Mb          0 b     216.00 Kb          7.18 Mb    1.08 GFLOPs    1.08 GFLOPs
  call_module  features_12_conv_0_1      43.08 Mb       43.07 Mb          0 b       4.50 Kb         43.07 Mb   56.45 MFLOPs   56.45 MFLOPs
  call_module  features_12_conv_0_2      43.07 Mb       43.07 Mb          0 b           0 b         43.07 Mb   11.29 MFLOPs        0 FLOPs
  call_module  features_12_conv_1_0      43.09 Mb       43.07 Mb          0 b      20.25 Kb         43.07 Mb  101.61 MFLOPs   58.53 GFLOPs
  call_module  features_12_conv_1_1      43.08 Mb       43.07 Mb          0 b       4.50 Kb         43.07 Mb   56.45 MFLOPs   56.45 MFLOPs
  call_module  features_12_conv_1_2      43.07 Mb       43.07 Mb          0 b           0 b         43.07 Mb   11.29 MFLOPs        0 FLOPs
  call_module    features_12_conv_2       7.39 Mb        7.18 Mb          0 b     216.00 Kb         43.07 Mb    1.08 GFLOPs    1.08 GFLOPs
  call_module    features_12_conv_3       1.88 Kb            0 b      7.18 Mb         768 b          7.18 Mb    9.41 MFLOPs    9.41 MFLOPs
call_function                 add_6       7.18 Mb        7.18 Mb          0 b           0 b         14.36 Mb    1.88 MFLOPs        0 FLOPs
  call_module  features_13_conv_0_0      43.28 Mb       43.07 Mb          0 b     216.00 Kb          7.18 Mb    1.08 GFLOPs    1.08 GFLOPs
  call_module  features_13_conv_0_1      43.08 Mb       43.07 Mb          0 b       4.50 Kb         43.07 Mb   56.45 MFLOPs   56.45 MFLOPs
  call_module  features_13_conv_0_2      43.07 Mb       43.07 Mb          0 b           0 b         43.07 Mb   11.29 MFLOPs        0 FLOPs
  call_module  features_13_conv_1_0      43.09 Mb       43.07 Mb          0 b      20.25 Kb         43.07 Mb  101.61 MFLOPs   58.53 GFLOPs
  call_module  features_13_conv_1_1      43.08 Mb       43.07 Mb          0 b       4.50 Kb         43.07 Mb   56.45 MFLOPs   56.45 MFLOPs
  call_module  features_13_conv_1_2      43.07 Mb       43.07 Mb          0 b           0 b         43.07 Mb   11.29 MFLOPs        0 FLOPs
  call_module    features_13_conv_2       7.39 Mb        7.18 Mb          0 b     216.00 Kb         43.07 Mb    1.08 GFLOPs    1.08 GFLOPs
  call_module    features_13_conv_3       1.88 Kb            0 b      7.18 Mb         768 b          7.18 Mb    9.41 MFLOPs    9.41 MFLOPs
call_function                 add_7       7.18 Mb        7.18 Mb          0 b           0 b         14.36 Mb    1.88 MFLOPs        0 FLOPs
  call_module  features_14_conv_0_0      43.28 Mb       43.07 Mb          0 b     216.00 Kb          7.18 Mb    1.08 GFLOPs    1.08 GFLOPs
  call_module  features_14_conv_0_1      43.08 Mb       43.07 Mb          0 b       4.50 Kb         43.07 Mb   56.45 MFLOPs   56.45 MFLOPs
  call_module  features_14_conv_0_2      43.07 Mb       43.07 Mb          0 b           0 b         43.07 Mb   11.29 MFLOPs        0 FLOPs
  call_module  features_14_conv_1_0      10.79 Mb       10.77 Mb          0 b      20.25 Kb         43.07 Mb   25.40 MFLOPs   14.63 GFLOPs
  call_module  features_14_conv_1_1      10.78 Mb       10.77 Mb          0 b       4.50 Kb         10.77 Mb   14.11 MFLOPs   14.11 MFLOPs
  call_module  features_14_conv_1_2      10.77 Mb       10.77 Mb          0 b           0 b         10.77 Mb    2.82 MFLOPs        0 FLOPs
  call_module    features_14_conv_2       3.34 Mb        2.99 Mb          0 b     360.00 Kb         10.77 Mb  451.58 MFLOPs  451.58 MFLOPs
  call_module    features_14_conv_3       2.99 Mb        2.99 Mb          0 b       1.25 Kb          2.99 Mb    3.92 MFLOPs    3.92 MFLOPs
  call_module  features_15_conv_0_0      18.53 Mb       17.94 Mb          0 b     600.00 Kb          2.99 Mb  752.64 MFLOPs  752.64 MFLOPs
  call_module  features_15_conv_0_1      17.96 Mb       17.94 Mb          0 b       7.50 Kb         17.94 Mb   23.52 MFLOPs   23.52 MFLOPs
  call_module  features_15_conv_0_2      17.94 Mb       17.94 Mb          0 b           0 b         17.94 Mb    4.70 MFLOPs        0 FLOPs
  call_module  features_15_conv_1_0      17.98 Mb       17.94 Mb          0 b      33.75 Kb         17.94 Mb   42.34 MFLOPs   40.64 GFLOPs
  call_module  features_15_conv_1_1      17.96 Mb       17.94 Mb          0 b       7.50 Kb         17.94 Mb   23.52 MFLOPs   23.52 MFLOPs
  call_module  features_15_conv_1_2      17.94 Mb       17.94 Mb          0 b           0 b         17.94 Mb    4.70 MFLOPs        0 FLOPs
  call_module    features_15_conv_2       3.58 Mb        2.99 Mb          0 b     600.00 Kb         17.94 Mb  752.64 MFLOPs  752.64 MFLOPs
  call_module    features_15_conv_3       3.12 Kb            0 b      2.99 Mb       1.25 Kb          2.99 Mb    3.92 MFLOPs    3.92 MFLOPs
call_function                 add_8       2.99 Mb        2.99 Mb          0 b           0 b          5.98 Mb  784.00 kFLOPs        0 FLOPs
  call_module  features_16_conv_0_0      18.53 Mb       17.94 Mb          0 b     600.00 Kb          2.99 Mb  752.64 MFLOPs  752.64 MFLOPs
  call_module  features_16_conv_0_1      17.96 Mb       17.94 Mb          0 b       7.50 Kb         17.94 Mb   23.52 MFLOPs   23.52 MFLOPs
  call_module  features_16_conv_0_2      17.94 Mb       17.94 Mb          0 b           0 b         17.94 Mb    4.70 MFLOPs        0 FLOPs
  call_module  features_16_conv_1_0      17.98 Mb       17.94 Mb          0 b      33.75 Kb         17.94 Mb   42.34 MFLOPs   40.64 GFLOPs
  call_module  features_16_conv_1_1      17.96 Mb       17.94 Mb          0 b       7.50 Kb         17.94 Mb   23.52 MFLOPs   23.52 MFLOPs
  call_module  features_16_conv_1_2      17.94 Mb       17.94 Mb          0 b           0 b         17.94 Mb    4.70 MFLOPs        0 FLOPs
  call_module    features_16_conv_2       3.58 Mb        2.99 Mb          0 b     600.00 Kb         17.94 Mb  752.64 MFLOPs  752.64 MFLOPs
  call_module    features_16_conv_3       3.12 Kb            0 b      2.99 Mb       1.25 Kb          2.99 Mb    3.92 MFLOPs    3.92 MFLOPs
call_function                 add_9       2.99 Mb        2.99 Mb          0 b           0 b          5.98 Mb  784.00 kFLOPs        0 FLOPs
  call_module  features_17_conv_0_0      18.53 Mb       17.94 Mb          0 b     600.00 Kb          2.99 Mb  752.64 MFLOPs  752.64 MFLOPs
  call_module  features_17_conv_0_1      17.96 Mb       17.94 Mb          0 b       7.50 Kb         17.94 Mb   23.52 MFLOPs   23.52 MFLOPs
  call_module  features_17_conv_0_2      17.94 Mb       17.94 Mb          0 b           0 b         17.94 Mb    4.70 MFLOPs        0 FLOPs
  call_module  features_17_conv_1_0      17.98 Mb       17.94 Mb          0 b      33.75 Kb         17.94 Mb   42.34 MFLOPs   40.64 GFLOPs
  call_module  features_17_conv_1_1      17.96 Mb       17.94 Mb          0 b       7.50 Kb         17.94 Mb   23.52 MFLOPs   23.52 MFLOPs
  call_module  features_17_conv_1_2      17.94 Mb       17.94 Mb          0 b           0 b         17.94 Mb    4.70 MFLOPs        0 FLOPs
  call_module    features_17_conv_2       7.15 Mb        5.98 Mb          0 b       1.17 Mb         17.94 Mb    1.51 GFLOPs    1.51 GFLOPs
  call_module    features_17_conv_3       5.99 Mb        5.98 Mb          0 b       2.50 Kb          5.98 Mb    7.84 MFLOPs    7.84 MFLOPs
  call_module         features_18_0      25.49 Mb       23.93 Mb          0 b       1.56 Mb          5.98 Mb    2.01 GFLOPs    2.01 GFLOPs
  call_module         features_18_1      25.00 Kb            0 b     23.93 Mb      10.00 Kb         23.93 Mb   31.36 MFLOPs   31.36 MFLOPs
  call_module         features_18_2           0 b            0 b     23.93 Mb           0 b         23.93 Mb    6.27 MFLOPs        0 FLOPs
call_function   adaptive_avg_pool2d           0 b            0 b    500.00 Kb           0 b         23.93 Mb    6.27 MFLOPs        0 FLOPs
call_function               flatten           0 b            0 b    500.00 Kb           0 b        500.00 Kb        0 FLOPs        0 FLOPs
  call_module          classifier_0     500.00 Kb      500.00 Kb          0 b           0 b        500.00 Kb  384.00 kFLOPs        0 FLOPs
  call_module          classifier_1           0 b            0 b    390.62 Kb       4.89 Mb        500.00 Kb  128.00 MFLOPs  128.10 MFLOPs
       output                output           0 b            0 b    390.62 Kb           0 b        390.62 Kb        0 FLOPs        0 FLOPs
```

![image](https://user-images.githubusercontent.com/78588128/211300536-bf78bda4-1ec3-4b96-8f00-e067e5c6f343.png)
