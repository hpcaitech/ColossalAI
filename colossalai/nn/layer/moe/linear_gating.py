import torch


class ForceFP32Parameter(torch.nn.Parameter):

    def half(self, memory_format=None):
        return self


class LinearGate(torch.nn.Linear):
    """Gate module used in MOE layer. Just a linear function without bias.
    But it should be kept as fp32 forever.
    """

    def __init__(self, d_model: int, num_experts: int, init_scale: float = 0.1):
        super().__init__(d_model, num_experts, bias=False)
        self.weight.data = ForceFP32Parameter(self.weight)


if __name__ == "__main__":
    a = torch.nn.Parameter(torch.randn(2, 3))
    a = ForceFP32Parameter(a)
    a = a.half()
    print(a)

    lingate = LinearGate(2, 3)

    b = lingate(torch.randn(1, 2))
    print(b)
