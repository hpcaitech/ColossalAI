from copy import deepcopy

from torch.testing import assert_close

from colossalai.elixir.kernels.attn_wrapper import wrap_attention
from tests.test_elixir.utils import TEST_MODELS, to_cuda


def exam_one_model(model_fn, data_fn):
    torch_model = model_fn().cuda()
    test_model = deepcopy(torch_model)
    test_model = wrap_attention(test_model)

    data = to_cuda(data_fn())
    torch_out = torch_model(**data)
    torch_out.backward()

    test_out = test_model(**data)
    test_out.backward()

    assert_close(torch_out, test_out)
    for (name, p_torch), p_test in zip(torch_model.named_parameters(), test_model.parameters()):
        assert_close(p_torch.grad, p_test.grad)


def test_gpt_atten_kernel():
    exam_one_model(*TEST_MODELS.get('gpt2_micro'))
    exam_one_model(*TEST_MODELS.get('opt_micro'))


if __name__ == '__main__':
    test_gpt_atten_kernel()
