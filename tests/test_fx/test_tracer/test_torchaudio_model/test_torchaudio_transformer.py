import torch
from torchaudio_utils import trace_and_compare
from torchaudio.models import Emformer, Conformer
import pytest


def test_conformer():
    input_dim = 80
    batch_size = 10
    num_frames = 400
    num_heads = 4
    ffn_dim = 128
    num_layers = 4
    depthwise_conv_kernel_size = 31

    model = Conformer(
        input_dim=input_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        depthwise_conv_kernel_size=depthwise_conv_kernel_size,
    )

    def data_gen():
        lengths = torch.randint(1, num_frames, (batch_size,))
        input = torch.rand(batch_size, int(lengths.max()), input_dim)
        return dict(input=input, lengths=lengths)

    def kwargs_transform(data):
        new_data = {}

        for k, v in data.items():
            new_data[f'{k}_1'] = v
        return new_data

    trace_and_compare(model, data_gen, need_meta=False, need_concrete=True, kwargs_transform=kwargs_transform)


@pytest.mark.skip("Tracing failed")
def test_emformer():
    input_dim = 128
    batch_size = 10
    num_heads = 8
    ffn_dim = 256
    num_layers = 3
    segment_length = 4
    num_frames = 400
    right_context_length = 1

    model = Emformer(input_dim, num_heads, ffn_dim, num_layers, segment_length, right_context_length)

    def data_gen():
        lengths = torch.randint(1, num_frames, (batch_size,))
        input = torch.rand(batch_size, num_frames, input_dim)
        return dict(input=input, lengths=lengths)

    trace_and_compare(model, data_gen, need_meta=True, need_concrete=False)


@pytest.mark.skip
def test_torchaudio_transformers():
    test_conformer()
    test_emformer()


if __name__ == "__main__":
    test_torchaudio_transformers()
