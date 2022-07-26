import torch
import torchaudio.models as tm
from colossalai.fx import ColoTracer
from torch.fx import GraphModule, Tracer
import pytest
from utils import trace_and_compare
    
    
def test_torchaudio_models():
    # added patch for conv_transpose
#     trace_and_compare(
#         module=tm.ConvTasNet(), 
#         data={'input': torch.rand(3, 1, 10)},
#         concrete_args={},
#         meta_args={'input': torch.rand(3, 1, 10, device='meta')},
#     )
    
#     trace_and_compare(
#         module=tm.DeepSpeech(100),
#         data={'x': torch.rand(3, 1, 20, 100)},
#         concrete_args={},
#         meta_args={}
#     )
    
#     trace_and_compare(
#         module=tm.Wav2Letter(),
#         data={'x': torch.rand(3, 1, 1000)},
#         concrete_args={},
#         meta_args={},
#     )
    
#     # added patch for GRU and RNN
#     trace_and_compare(
#         module=tm.WaveRNN(upsample_scales=[5,5,8], n_classes=512, hop_length=200),
#         data={
#             'waveform': torch.rand(3, 1, 1200), 
#             # shape: (n_batch, 1, (n_time - kernel_size + 1) * hop_length)
#             'specgram': torch.rand(3, 1, 128, 10),
#             # shape: (n_batch, 1, n_freq, n_time)
#         },
#         concrete_args={},
#         meta_args={
#             'waveform': torch.rand(3, 1, 1200, device='meta'), 
#             # shape: (n_batch, 1, (n_time - kernel_size + 1) * hop_length)
#             'specgram': torch.rand(3, 1, 128, 10, device='meta'),
#             # shape: (n_batch, 1, n_freq, n_time)
#         },
#     )
    
    # warning occurs
    lengths = torch.randint(1, 40, (10,))
    trace_and_compare(
        module=tm.Conformer(input_dim=10, num_heads=2, ffn_dim=128, num_layers=4, 
                          depthwise_conv_kernel_size=31),
        data= {
            'input': torch.rand(10, int(lengths.max()), 10),  # (batch, num_frames, input_dim)
            'lengths': lengths,
        },
        concrete_args = {'lengths': lengths},
        meta_args={},
    )
    
    token_lengths = torch.randint(1, 40, (10,))
    ma
    trace_and_compare(
        module=tm.Tacotron2()
        data= {
            'input': torch.rand(10, int(lengths.max()), 10),  # (batch, num_frames, input_dim)
            'lengths': lengths,
        },
        concrete_args = {'lengths': lengths},
        meta_args={},
    )
        


if __name__ == "__main__":
    test_torchaudio_models()