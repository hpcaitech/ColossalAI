# Analyzer

# Overview
The Analyzer is a collection of static graph utils including Colossal-AI FX. Features include:
- MetaTensor -- enabling:
  - Ahead-of-time Profiling
  - Shape Propagation
  - Ideal Flop Counter
- symbolic_trace()
  - Robust Control-flow Tracing / Recompile
  - Robust Activation Checkpoint Tracing / CodeGen
  - Easy-to-define Bias-Addition Split
- symbolic_profile()
  - Support ``MetaTensorMode``, where all Tensor operations are executed symbolically.
  - Shape Inference Across Device and Unified ``MetaInfo``
  - Ideal Flop Counter https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505

# Quickstart
## Analyzer.FX
**Reference:**

  https://pytorch.org/docs/stable/fx.html [[paper](https://arxiv.org/pdf/2112.08429)]


torch.FX is a toolkit for developers to use to transform nn.Module instances. FX consists of three main components: a symbolic tracer, an intermediate representation, and Python code generation. FX.Tracer hacks _\_\_torch_function\_\__ and use a Proxy object to propagate through any forward function of torch.nn.Module.
![image](https://user-images.githubusercontent.com/78588128/212531495-bbb934dd-dbbb-4578-8869-6171973f7dd8.png)
ColossalAI FX is modified from torch.FX, with the extra capability of ahead-of-time profiling enabled by the subclass of ``MetaTensor``.

### Analyzer.FX.symbolic_trace()
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

### Analyzer.FX.symbolic_profile()

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

Meanwhile, it is easy to keep track of the memory usage of each node when doing shape propagation. However, the drawbacks of FX is that not every ``call_function`` saves its input for backward, and different tensor that flows within one FX.Graph can actually have the same layout. This raises problems for fine-grained profiling.

![image](https://user-images.githubusercontent.com/78588128/215312957-7eb6cbc3-61b2-49cf-95a4-6b859149eb8d.png)

To address this problem, I came up with a simulated environment enabled by ``torch.autograd.graph.saved_tensor_hooks`` and fake ``data_ptr`` (check ``_subclasses/meta_tensor.py`` for more details of ``data_ptr`` updates).

```python
class sim_env(saved_tensors_hooks):
    """
    A simulation of memory allocation and deallocation in the forward pass
    using ``saved_tensor_hooks``.

    Attributes:
        ctx (Dict[int, torch.Tensor]): A dictionary that maps the
            data pointer of a tensor to the tensor itself. This is used
            to track the memory allocation and deallocation.

        param_ctx (Dict[int, torch.Tensor]): A dictionary that maps the
            data pointer of all model parameters to the parameter itself.
            This avoids overestimating the memory usage of the intermediate activations.
    """

    def __init__(self, module: Optional[torch.nn.Module] = None):
        super().__init__(self.pack_hook, self.unpack_hook)
        self.ctx = {}
        self.param_ctx = {param.data_ptr(): param for param in module.parameters()}
        self.buffer_ctx = {buffer.data_ptr(): buffer for buffer in module.buffers()} if module else {}

    def pack_hook(self, tensor: torch.Tensor):
        if tensor.data_ptr() not in self.param_ctx and tensor.data_ptr() not in self.buffer_ctx:
            self.ctx[tensor.data_ptr()] = tensor
        return tensor

    def unpack_hook(self, tensor):
        return tensor
```
The ``ctx`` variable will keep track of all saved tensors with a unique identifier. It is likely that ``nn.Parameter`` is also counted in the ``ctx``, which is not desired. To avoid this, we can use ``param_ctx`` to keep track of all parameters in the model. The ``buffer_ctx`` is used to keep track of all buffers in the model. The ``local_ctx`` that is attached to each ``Node`` marks the memory usage of the stage to which the node belongs. With simple ``intersect``, ``union`` and ``subtract`` operations, we can get any memory-related information. For non-profileable nodes, you might add your customized profile rules to simulate the memory allocation. If a ``Graph`` is modified with some non-PyTorch functions, such as fused operands, you can register the shape propagation rule with the decorator.

```python
@register_shape_impl(fuse_conv_bn)
def fuse_conv_bn_shape_impl(*args, **kwargs):
     # infer output shape here
     return torch.empty(output_shape, device=output_device)
```

An important notice is that ``ShapeProp`` will attach additional information to the graph, which will be exactly the input of ``Profiler``.

#### GraphProfiler
``GraphProfiler`` executes at the node level, and profiles both forward and backward within one node. For example, ``FlopProfiler`` will profile the forward and backward FLOPs of a node, and ``CommunicationProfiler`` will profile the forward and backward communication cost of a node. The ``GraphProfiler`` will attach the profiling results to the ``Node``. These procedures are decoupled for better extensibility.

To provide a general insight of the profiled results, you can set ``verbose=True`` to print the summary as well.
```python
model = tm.resnet18()
sample = torch.rand(100, 3, 224, 224)
meta_args = dict(x=sample)
gm = symbolic_trace(model, meta_args=meta_args)
gm = symbolic_profile(gm, sample, verbose=True)

============================================================ Results =====================================================================
       Op type                                              Op    Accumulate size    Incremental size    Output size    Temp size    Param size    Backward size      Fwd FLOPs      Bwd FLOPs
-------------  ----------------------------------------------  -----------------  ------------------  -------------  -----------  ------------  ---------------  -------------  -------------
  placeholder                                               x            4.59 Mb                 0 b        4.59 Mb          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module                                       conv_proj            4.59 Mb                 0 b            0 b      4.59 Mb       2.25 Mb          4.59 Mb  924.84 MFLOPs  924.84 MFLOPs
  call_method                                         reshape            4.59 Mb                 0 b            0 b      4.59 Mb           0 b          4.59 Mb        0 FLOPs        0 FLOPs
  call_method                                         permute            4.59 Mb                 0 b            0 b      4.59 Mb           0 b          4.59 Mb        0 FLOPs        0 FLOPs
     get_attr                                     class_token            4.59 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_method                                          expand            4.59 Mb                 0 b            0 b     24.00 Kb       3.00 Kb              0 b        0 FLOPs    6.14 kFLOPs
call_function                                             cat            4.59 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
     get_attr                           encoder_pos_embedding            4.59 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                             add            9.21 Mb             4.62 Mb        4.62 Mb          0 b     591.00 Kb          4.62 Mb    1.21 MFLOPs    1.21 MFLOPs
  call_module                                 encoder_dropout            9.21 Mb                 0 b        4.62 Mb          0 b           0 b          4.62 Mb        0 FLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_0_ln_1            9.22 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module   encoder_layers_encoder_layer_0_self_attention           46.52 Mb            37.30 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                         getitem           46.52 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                       getitem_1           46.52 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module          encoder_layers_encoder_layer_0_dropout           46.52 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                           add_1           51.14 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_0_ln_2           51.15 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module            encoder_layers_encoder_layer_0_mlp_0           74.24 Mb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_0_mlp_1           92.71 Mb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module            encoder_layers_encoder_layer_0_mlp_2           92.71 Mb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_0_mlp_3           92.71 Mb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_0_mlp_4           92.71 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                           add_2           97.32 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_1_ln_1          101.95 Mb             4.63 Mb        4.62 Mb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module   encoder_layers_encoder_layer_1_self_attention          134.63 Mb            32.68 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                       getitem_2          134.63 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                       getitem_3          134.63 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module          encoder_layers_encoder_layer_1_dropout          134.63 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                           add_3          139.25 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_1_ln_2          139.26 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module            encoder_layers_encoder_layer_1_mlp_0          162.35 Mb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_1_mlp_1          180.82 Mb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module            encoder_layers_encoder_layer_1_mlp_2          180.82 Mb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_1_mlp_3          180.82 Mb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_1_mlp_4          180.82 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                           add_4          185.43 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_2_ln_1          190.06 Mb             4.63 Mb        4.62 Mb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module   encoder_layers_encoder_layer_2_self_attention          222.74 Mb            32.68 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                       getitem_4          222.74 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                       getitem_5          222.74 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module          encoder_layers_encoder_layer_2_dropout          222.74 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                           add_5          227.36 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_2_ln_2          227.37 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module            encoder_layers_encoder_layer_2_mlp_0          250.46 Mb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_2_mlp_1          268.93 Mb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module            encoder_layers_encoder_layer_2_mlp_2          268.93 Mb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_2_mlp_3          268.93 Mb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_2_mlp_4          268.93 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                           add_6          273.54 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_3_ln_1          278.17 Mb             4.63 Mb        4.62 Mb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module   encoder_layers_encoder_layer_3_self_attention          310.86 Mb            32.68 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                       getitem_6          310.86 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                       getitem_7          310.86 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module          encoder_layers_encoder_layer_3_dropout          310.86 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                           add_7          315.47 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_3_ln_2          315.48 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module            encoder_layers_encoder_layer_3_mlp_0          338.57 Mb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_3_mlp_1          357.04 Mb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module            encoder_layers_encoder_layer_3_mlp_2          357.04 Mb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_3_mlp_3          357.04 Mb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_3_mlp_4          357.04 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                           add_8          361.66 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_4_ln_1          366.29 Mb             4.63 Mb        4.62 Mb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module   encoder_layers_encoder_layer_4_self_attention          398.97 Mb            32.68 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                       getitem_8          398.97 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                       getitem_9          398.97 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module          encoder_layers_encoder_layer_4_dropout          398.97 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                           add_9          403.58 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_4_ln_2          403.60 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module            encoder_layers_encoder_layer_4_mlp_0          426.68 Mb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_4_mlp_1          445.15 Mb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module            encoder_layers_encoder_layer_4_mlp_2          445.15 Mb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_4_mlp_3          445.15 Mb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_4_mlp_4          445.15 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_10          449.77 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_5_ln_1          454.40 Mb             4.63 Mb        4.62 Mb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module   encoder_layers_encoder_layer_5_self_attention          487.08 Mb            32.68 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                      getitem_10          487.08 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                      getitem_11          487.08 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module          encoder_layers_encoder_layer_5_dropout          487.08 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_11          491.70 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_5_ln_2          491.71 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module            encoder_layers_encoder_layer_5_mlp_0          514.79 Mb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_5_mlp_1          533.26 Mb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module            encoder_layers_encoder_layer_5_mlp_2          533.26 Mb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_5_mlp_3          533.26 Mb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_5_mlp_4          533.26 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_12          537.88 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_6_ln_1          542.51 Mb             4.63 Mb        4.62 Mb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module   encoder_layers_encoder_layer_6_self_attention          575.19 Mb            32.68 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                      getitem_12          575.19 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                      getitem_13          575.19 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module          encoder_layers_encoder_layer_6_dropout          575.19 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_13          579.81 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_6_ln_2          579.82 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module            encoder_layers_encoder_layer_6_mlp_0          602.90 Mb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_6_mlp_1          621.37 Mb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module            encoder_layers_encoder_layer_6_mlp_2          621.37 Mb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_6_mlp_3          621.37 Mb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_6_mlp_4          621.37 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_14          625.99 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_7_ln_1          630.62 Mb             4.63 Mb        4.62 Mb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module   encoder_layers_encoder_layer_7_self_attention          663.30 Mb            32.68 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                      getitem_14          663.30 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                      getitem_15          663.30 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module          encoder_layers_encoder_layer_7_dropout          663.30 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_15          667.92 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_7_ln_2          667.93 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module            encoder_layers_encoder_layer_7_mlp_0          691.02 Mb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_7_mlp_1          709.48 Mb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module            encoder_layers_encoder_layer_7_mlp_2          709.48 Mb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_7_mlp_3          709.48 Mb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_7_mlp_4          709.48 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_16          714.10 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_8_ln_1          718.73 Mb             4.63 Mb        4.62 Mb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module   encoder_layers_encoder_layer_8_self_attention          751.41 Mb            32.68 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                      getitem_16          751.41 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                      getitem_17          751.41 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module          encoder_layers_encoder_layer_8_dropout          751.41 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_17          756.03 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_8_ln_2          756.04 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module            encoder_layers_encoder_layer_8_mlp_0          779.13 Mb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_8_mlp_1          797.60 Mb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module            encoder_layers_encoder_layer_8_mlp_2          797.60 Mb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_8_mlp_3          797.60 Mb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_8_mlp_4          797.60 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_18          802.21 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_9_ln_1          806.84 Mb             4.63 Mb        4.62 Mb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module   encoder_layers_encoder_layer_9_self_attention          839.52 Mb            32.68 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                      getitem_18          839.52 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                      getitem_19          839.52 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module          encoder_layers_encoder_layer_9_dropout          839.52 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_19          844.14 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module             encoder_layers_encoder_layer_9_ln_2          844.15 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module            encoder_layers_encoder_layer_9_mlp_0          867.24 Mb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_9_mlp_1          885.71 Mb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module            encoder_layers_encoder_layer_9_mlp_2          885.71 Mb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_9_mlp_3          885.71 Mb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module            encoder_layers_encoder_layer_9_mlp_4          885.71 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_20          890.32 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_10_ln_1          894.95 Mb             4.63 Mb        4.62 Mb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module  encoder_layers_encoder_layer_10_self_attention          927.63 Mb            32.68 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                      getitem_20          927.63 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                      getitem_21          927.63 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module         encoder_layers_encoder_layer_10_dropout          927.63 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_21          932.25 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_10_ln_2          932.26 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module           encoder_layers_encoder_layer_10_mlp_0          955.35 Mb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module           encoder_layers_encoder_layer_10_mlp_1          973.82 Mb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module           encoder_layers_encoder_layer_10_mlp_2          973.82 Mb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module           encoder_layers_encoder_layer_10_mlp_3          973.82 Mb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module           encoder_layers_encoder_layer_10_mlp_4          973.82 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_22          978.44 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_11_ln_1          983.06 Mb             4.63 Mb        4.62 Mb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module  encoder_layers_encoder_layer_11_self_attention         1015.75 Mb            32.68 Mb            0 b      4.62 Mb       9.01 Mb         13.85 Mb    4.20 GFLOPs    8.40 GFLOPs
call_function                                      getitem_22         1015.75 Mb                 0 b            0 b      4.62 Mb           0 b              0 b        0 FLOPs        0 FLOPs
call_function                                      getitem_23         1015.75 Mb                 0 b            0 b          0 b           0 b              0 b        0 FLOPs        0 FLOPs
  call_module         encoder_layers_encoder_layer_11_dropout         1015.75 Mb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_23         1020.36 Mb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module            encoder_layers_encoder_layer_11_ln_2         1020.38 Mb            12.31 Kb            0 b      4.62 Mb       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
  call_module           encoder_layers_encoder_layer_11_mlp_0            1.02 Gb            23.09 Mb       18.47 Mb          0 b       9.01 Mb          4.62 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module           encoder_layers_encoder_layer_11_mlp_1            1.04 Gb            18.47 Mb       18.47 Mb          0 b           0 b         18.47 Mb    4.84 MFLOPs    4.84 MFLOPs
  call_module           encoder_layers_encoder_layer_11_mlp_2            1.04 Gb                 0 b       18.47 Mb          0 b           0 b         18.47 Mb        0 FLOPs        0 FLOPs
  call_module           encoder_layers_encoder_layer_11_mlp_3            1.04 Gb                 0 b            0 b      4.62 Mb       9.00 Mb         18.47 Mb    3.72 GFLOPs    7.44 GFLOPs
  call_module           encoder_layers_encoder_layer_11_mlp_4            1.04 Gb                 0 b            0 b      4.62 Mb           0 b          4.62 Mb        0 FLOPs        0 FLOPs
call_function                                          add_24            1.04 Gb             4.62 Mb        4.62 Mb          0 b           0 b          9.23 Mb    1.21 MFLOPs        0 FLOPs
  call_module                                      encoder_ln            1.04 Gb            36.31 Kb       24.00 Kb          0 b       6.00 Kb          4.62 Mb    6.05 MFLOPs    6.05 MFLOPs
call_function                                      getitem_24            1.04 Gb                 0 b       24.00 Kb          0 b           0 b          4.62 Mb        0 FLOPs        0 FLOPs
  call_module                                      heads_head            1.04 Gb                 0 b            0 b     31.25 Kb       2.93 Mb         24.00 Kb    6.14 MFLOPs   12.30 MFLOPs
       output                                          output            1.04 Gb                 0 b            0 b     31.25 Kb           0 b         31.25 Kb        0 FLOPs        0 FLOPs
```
