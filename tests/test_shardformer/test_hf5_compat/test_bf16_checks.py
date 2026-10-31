import pytest
import torch

from tests.test_shardformer.test_hf5_compat.bf16_checks import compare_parameters, compare_tensor


def test_finite_and_threshold():
    with pytest.raises(AssertionError):
        compare_tensor(torch.ones(2), torch.tensor([1.0, float("nan")]))
    with pytest.raises(AssertionError):
        compare_tensor(torch.zeros(2), torch.tensor([0.0, 0.02]))


def test_missing_gradients_and_parameters():
    reference = torch.nn.Linear(2, 2)
    actual = torch.nn.Linear(2, 2)
    actual.load_state_dict(reference.state_dict())
    reference(torch.ones(1, 2)).sum().backward()
    with pytest.raises(AssertionError):
        compare_parameters(reference, actual)
    actual(torch.ones(1, 2)).sum().backward()
    assert len(compare_parameters(reference, actual)) == 2
    with pytest.raises(AssertionError):
        compare_parameters(reference, torch.nn.Linear(2, 2, bias=False), gradients=False)


def test_unexplained_shape_mismatch():
    reference = torch.nn.Linear(2, 4, bias=False)
    actual = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(AssertionError):
        compare_parameters(reference, actual, gradients=False)
