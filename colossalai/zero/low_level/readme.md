# Low Level ZeRO
>Low Level ZeRO == ZeRO-DP stage 1 and 2, we would denote it as ZeRO.
## Examples of ZeRO and gradient accumulation

The code below only shows a typical gradient accumulation process, and it drops a lot of details, such as the processing of loss.

```python
# examples of ZeRO1 with gradient accumulation
...
outputs = model(input)
loss = SomeLoss(outputs)
if (idx + 1) % ACCUMULATE_STEP != 0:
    with booster.no_sync(model, optimizer):
        # under this context, the gradient would not sync when backward,
        # left each rank having different gradient.
        # It saves the backward time
        booster.backward(loss, optimizer)
        continue
else:
    # need to sync all the accumulated gradient
    booster.backward(loss, optimizer):
    optimizer.step()
    ...
```

```python
# example of ZeRO2 with gradient accumulation

...
outputs = model(input)
loss = SomeLoss(outputs)
# ZeRO2 split the gradients and can NOT accumulate gradient with syncing.
booster.backward(loss, optimizer)
if (idx + 1) % ACCUMULATE_STEP == 0:
    optimizer.step()
...
```


## Design:
### Notion
`p32` denotes the param copy in the optimizer
`p` denotes the model param
`g` denotes the gradient

### INIT
In low level zero(1, 2), `p32` is split. Different from the previous implement, we split each `p32` evenly by world_size. Thus, rank0 got a param list as `[p00, p10]`, rank1 got a param list as `[p-01, p-11]`, etc.
<img width="840" alt="image" src="https://github.com/hpcaitech/ColossalAI/assets/74758262/f7758d7d-c5e5-44a4-a121-3aba8b05c904">

For the detailed implementation, we first pad `p` for it can be split by world_size if needed. Then, we would view it to the shape `[world_size, -1]`, and each rank got its own part `p32` by cloning.

### BWD
To leverage the communication, a gradient would be added to a bucket first. When the bucket is full, each `g` in it would be reshaped as `[world_size, -1]`. And the `[local_rank]` parts would be united.
The data structure looks like this:
```
{
0: [g-00, g-10],
1: [g-01, g-11],
2: [g-02, g-12]
}
```
After that, the gradients would be flattened by rank, and the data structure looks like this:
```
# g-X0 means flatten([g-00, g-10])
{
0: [g-X0],
1: [g-X1],
2: [g-X2]
}
```
For zero1, we iterate the dictionary and do `all_reduce`. For zero2, we can just do `reduce-scatter`.

### Optim
For each rank gets its own `p32` and the counterpart `g`, it is quite easy to do `optim.step()`.

However, we have to consider a situation of layer drop, for instance:
```
class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.drop_linear = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
```
And the solution is to build a mapping of `p32`, `p`, and `g`. Before `optim.step()`, we collect `p` which `requires_grad=True` and `p.grad != None` as a real working param. And select the counterpart `p32` and `g`.
