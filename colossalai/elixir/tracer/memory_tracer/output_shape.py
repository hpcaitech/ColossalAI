import torch


# Output functions come from https://github.com/pytorch/pytorch/blob/master/torch/_meta_registrations.py
def check_cuda_mm(*args):
    for x in args:
        assert isinstance(x, torch.Tensor)
        assert x.device.type == 'cuda'


def mm_output(a, b):
    assert a.dim() == 2, 'a must be 2D'
    assert b.dim() == 2, 'b must be 2D'
    N, M1 = a.shape
    M2, P = b.shape
    assert M1 == M2, 'a and b must have same reduction dim'
    return (N, P)


def addmm_output(bias, x, y):
    return mm_output(x, y)


def common_baddbmm_bmm(batch1, batch2, is_bmm, self_baddbmm=None):
    assert batch1.dim() == 3, 'batch1 must be a 3D tensor'
    assert batch2.dim() == 3, 'batch2 must be a 3D tensor'

    batch1_sizes = batch1.size()
    batch2_sizes = batch2.size()

    bs = batch1_sizes[0]
    contraction_size = batch1_sizes[2]
    res_rows = batch1_sizes[1]
    res_cols = batch2_sizes[2]
    output_size = (bs, res_rows, res_cols)

    assert batch2_sizes[0] == bs and batch2_sizes[1] == contraction_size

    if not is_bmm and self_baddbmm is not None:
        assert self_baddbmm.dim() == 3, 'self must be a 3D tensor'
        assert self_baddbmm.size() == output_size, \
            f'Expected an input tensor shape with shape {output_size} but got shape: {self_baddbmm.size()}'

    return output_size


def bmm_output(mat1, mat2):
    return common_baddbmm_bmm(mat1, mat2, True)
