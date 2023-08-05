# Low Level ZeRO
>Low Level ZeRO == ZeRO-DP stage 1 and 2, we would denote it as ZeRO.

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
# g-0 means flatten([g-00, g-10])
{
0: [g-0],
1: [g-1],
2: [g-2]
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
