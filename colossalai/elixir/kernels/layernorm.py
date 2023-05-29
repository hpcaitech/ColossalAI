from apex.normalization.fused_layer_norm import fused_layer_norm, fused_layer_norm_affine


def ln_func(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    if weight is None:
        assert bias is None
        return fused_layer_norm(input, normalized_shape, eps)
    else:
        assert weight is not None and bias is not None
        return fused_layer_norm_affine(input, weight, bias, normalized_shape, eps)
