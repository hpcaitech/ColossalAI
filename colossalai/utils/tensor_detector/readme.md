# Tensor Detector

This tool supports you to detect tensors on both CPU and GPU. However, there will always be some strange tensors on CPU, including the rng state of PyTorch.

## Example

An example is worth than a thousand words.

The code below defines a simple MLP module, with which we will show you how to use the tool.

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(64, 8),
                                 nn.ReLU(),
                                 nn.Linear(8, 32))
    def forward(self, x):
        return self.mlp(x)
```

And here is how to use the tool.

```python
from colossalai.utils import TensorDetector

# create random data
data = torch.rand(64, requires_grad=True).cuda()
data.retain_grad()
# create the module
model = MLP().cuda()
# create the detector
# by passing the model to the detector, it can distinguish module parameters from common tensors
detector = TensorDetector(include_cpu=False, module=model)
detector.detect()

out = model(data)

detector.detect()

loss = out.sum()
loss.backward()

detector.detect()
```

I have made some comments on the right of the output for your understanding.

Note that the total `Mem` of all the tensors and parameters is not equal to `Total GPU Memory Allocated`.  PyTorch's memory management is really complicated, and for models of a large scale, it's impossible to figure out clearly.

**The order of print is not equal to the order the tensor creates, but they are really close.**

```bash
------------------------------------------------------------------------------------------------------------
   Tensor                            device               shape      grad               dtype            Mem
------------------------------------------------------------------------------------------------------------
+  Tensor                            cuda:0               (64,)      True       torch.float32          256 B    # data
+  mlp.0.weight                      cuda:0             (8, 64)      True       torch.float32         2.0 KB
+  mlp.0.bias                        cuda:0                (8,)      True       torch.float32           32 B
+  mlp.2.weight                      cuda:0             (32, 8)      True       torch.float32         1.0 KB
+  mlp.2.bias                        cuda:0               (32,)      True       torch.float32          128 B
------------------------------------------------------------------------------------------------------------
Detect Location: "test_tensor_detector.py" line 27
Total GPU Memory Allocated on cuda:0 is 4.5 KB
------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------
   Tensor                            device               shape      grad               dtype            Mem
------------------------------------------------------------------------------------------------------------
+  Tensor                            cuda:0                (8,)      True       torch.float32           32 B    # activation
+  Tensor                            cuda:0               (32,)      True       torch.float32          128 B    # output
------------------------------------------------------------------------------------------------------------
Detect Location: "test_tensor_detector.py" line 30
Total GPU Memory Allocated on cuda:0 is 5.5 KB
------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------
   Tensor                            device               shape      grad               dtype            Mem
------------------------------------------------------------------------------------------------------------
+  Tensor                            cuda:0                  ()      True       torch.float32            4 B    # loss
------------------------------------------------------------------------------------------------------------
Detect Location: "test_tensor_detector.py" line 32
Total GPU Memory Allocated on cuda:0 is 6.0 KB
------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------
   Tensor                            device               shape      grad               dtype            Mem
------------------------------------------------------------------------------------------------------------
+  Tensor (with grad)                cuda:0               (64,)      True       torch.float32          512 B    # data with grad
+  mlp.0.weight (with grad)          cuda:0             (8, 64)      True       torch.float32         4.0 KB    # for use data.retain_grad()
+  mlp.0.bias (with grad)            cuda:0                (8,)      True       torch.float32           64 B
+  mlp.2.weight (with grad)          cuda:0             (32, 8)      True       torch.float32         2.0 KB
+  mlp.2.bias (with grad)            cuda:0               (32,)      True       torch.float32          256 B

-  mlp.0.weight                      cuda:0             (8, 64)      True       torch.float32         2.0 KB
-  mlp.0.bias                        cuda:0                (8,)      True       torch.float32           32 B
-  mlp.2.weight                      cuda:0             (32, 8)      True       torch.float32         1.0 KB
-  mlp.2.bias                        cuda:0               (32,)      True       torch.float32          128 B
-  Tensor                            cuda:0               (64,)      True       torch.float32          256 B
-  Tensor                            cuda:0                (8,)      True       torch.float32           32 B    # deleted activation
------------------------------------------------------------------------------------------------------------
Detect Location: "test_tensor_detector.py" line 34
Total GPU Memory Allocated on cuda:0 is 10.0 KB
------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------
   Tensor                            device               shape      grad               dtype            Mem
------------------------------------------------------------------------------------------------------------
+  Tensor                            cuda:0               (64,)     False       torch.float32          256 B
+  Tensor                            cuda:0             (8, 64)     False       torch.float32         2.0 KB
+  Tensor                            cuda:0                (8,)     False       torch.float32           32 B
+  Tensor                            cuda:0             (32, 8)     False       torch.float32         1.0 KB
+  Tensor                            cuda:0               (32,)     False       torch.float32          128 B
------------------------------------------------------------------------------------------------------------
Detect Location: "test_tensor_detector.py" line 36
Total GPU Memory Allocated on cuda:0 is 14.0 KB
------------------------------------------------------------------------------------------------------------
```

## Reference

 This tool was inspired by https://github.com/Stonesjtu/pytorch_memlab/blob/master/pytorch_memlab/mem_reporter.py
 and https://github.com/Oldpan/Pytorch-Memory-Utils
