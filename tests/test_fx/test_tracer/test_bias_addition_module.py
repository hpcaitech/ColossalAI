import torch

from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.testing import clear_cache_before_run


class LinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = x * 2

        return x


class ConvModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias
        )

    def forward(self, x):
        x = self.conv(x)
        x = x * 2

        return x


@clear_cache_before_run()
def test_linear_module():
    model = LinearModel(3, 6)
    tracer = ColoTracer()
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %linear_weight : [#users=1] = get_attr[target=linear.weight]
    #     %linear_bias : [#users=1] = get_attr[target=linear.bias]
    #     %linear : [#users=1] = call_function[target=torch._C._nn.linear](args = (%x, %linear_weight), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%linear, %linear_bias), kwargs = {})
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%add, 2), kwargs = {})
    #     return mul
    graph = tracer.trace(root=model, meta_args={"x": torch.rand(3, 3).to("meta")})
    # def forward(self, x : torch.Tensor):
    #     linear_weight = self.linear.weight
    #     linear_bias = self.linear.bias
    #     linear = torch._C._nn.linear(x, linear_weight);  x = linear_weight = None
    #     add = linear + linear_bias;  linear = linear_bias = None
    #     mul = add * 2;  add = None
    #     return mul
    gm = ColoGraphModule(model, graph)
    gm.recompile()
    node_list = list(graph.nodes)
    for node in node_list:
        if node.op == "output":
            continue
        assert hasattr(node, "_meta_data")
    weight_node = node_list[1]
    bias_node = node_list[2]
    linear_node = node_list[3]
    add_node = node_list[4]
    assert weight_node._meta_data.shape == (6, 3)
    assert bias_node._meta_data.shape == (6,)
    assert linear_node._meta_data.shape == (3, 6)
    assert add_node._meta_data.shape == (3, 6)


@clear_cache_before_run()
def test_conv_module():
    model = ConvModel(3, 6, 2)
    tracer = ColoTracer()
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %conv_weight : [#users=1] = get_attr[target=conv.weight]
    #     %conv_bias : [#users=1] = get_attr[target=conv.bias]
    #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%x, %conv_weight), kwargs = {})
    #     %view : [#users=1] = call_method[target=view](args = (%conv_bias, [1, -1, 1, 1]), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%conv2d, %view), kwargs = {})
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%add, 2), kwargs = {})
    #     return mul
    graph = tracer.trace(root=model, meta_args={"x": torch.rand(4, 3, 64, 64).to("meta")})
    # def forward(self, x : torch.Tensor):
    #     conv_weight = self.conv.weight
    #     conv_bias = self.conv.bias
    #     conv2d = torch.conv2d(x, conv_weight);  x = conv_weight = None
    #     view = conv_bias.view([1, -1, 1, 1]);  conv_bias = None
    #     add = conv2d + view;  conv2d = view = None
    #     mul = add * 2;  add = None
    #     return mul
    gm = ColoGraphModule(model, graph)

    gm.recompile()
    node_list = list(graph.nodes)
    for node in node_list:
        if node.op == "output":
            continue
        assert hasattr(node, "_meta_data")
    weight_node = node_list[1]
    bias_node = node_list[2]
    conv_node = node_list[3]
    view_node = node_list[4]
    add_node = node_list[5]
    assert weight_node._meta_data.shape == (6, 3, 2, 2)
    assert bias_node._meta_data.shape == (6,)
    assert conv_node._meta_data.shape == (4, 6, 63, 63)
    assert view_node._meta_data.shape == (6, 1, 1)
    assert add_node._meta_data.shape == (4, 6, 63, 63)


if __name__ == "__main__":
    test_linear_module()
    test_conv_module()
