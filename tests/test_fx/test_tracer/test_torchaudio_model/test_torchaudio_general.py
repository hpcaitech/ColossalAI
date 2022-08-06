import torch
from torchaudio_utils import trace_and_compare
from torchaudio.models import ConvTasNet, DeepSpeech, Wav2Letter, WaveRNN
from torchaudio.models.wavernn import MelResNet, UpsampleNetwork
import pytest


def test_wave2letter_waveform():
    batch_size = 2
    num_features = 1
    num_classes = 40
    input_length = 320

    model = Wav2Letter(num_classes=num_classes, num_features=num_features)

    def data_gen():
        x = torch.rand(batch_size, num_features, input_length)
        return dict(x=x)

    trace_and_compare(model, data_gen, need_meta=False, need_concrete=False)


def test_wave2letter_mfcc():
    batch_size = 2
    num_features = 13
    num_classes = 40
    input_length = 2

    model = Wav2Letter(num_classes=num_classes, input_type="mfcc", num_features=num_features)

    def data_gen():
        x = torch.rand(batch_size, num_features, input_length)
        return dict(x=x)

    trace_and_compare(model, data_gen, need_meta=False, need_concrete=False)


def test_melresnet_waveform():
    n_batch = 2
    n_time = 200
    n_freq = 100
    n_output = 128
    n_res_block = 10
    n_hidden = 128
    kernel_size = 5

    model = MelResNet(n_res_block, n_freq, n_hidden, n_output, kernel_size)

    def data_gen():
        x = torch.rand(n_batch, n_freq, n_time)
        return dict(specgram=x)

    trace_and_compare(model, data_gen, need_meta=False, need_concrete=False)


def test_upsample_network_waveform():
    upsample_scales = [5, 5, 8]
    n_batch = 2
    n_time = 200
    n_freq = 100
    n_output = 64
    n_res_block = 10
    n_hidden = 32
    kernel_size = 5

    total_scale = 1
    for upsample_scale in upsample_scales:
        total_scale *= upsample_scale

    model = UpsampleNetwork(upsample_scales, n_res_block, n_freq, n_hidden, n_output, kernel_size)

    def data_gen():
        x = torch.rand(n_batch, n_freq, n_time)
        return dict(specgram=x)

    trace_and_compare(model, data_gen, need_meta=False, need_concrete=False)


def test_wavernn_waveform():
    upsample_scales = [2, 2, 5]
    n_rnn = 16
    n_fc = 16
    n_classes = 10
    hop_length = 20
    n_batch = 2
    n_time = 20
    n_freq = 10
    n_output = 16
    n_res_block = 3
    n_hidden = 16
    kernel_size = 5

    model = WaveRNN(upsample_scales, n_classes, hop_length, n_res_block, n_rnn, n_fc, kernel_size, n_freq, n_hidden,
                    n_output)

    def data_gen():
        x = torch.rand(n_batch, 1, hop_length * (n_time - kernel_size + 1))
        mels = torch.rand(n_batch, 1, n_freq, n_time)
        return dict(waveform=x, specgram=mels)

    trace_and_compare(model, data_gen, need_meta=True, need_concrete=False)


def test_convtasnet_config():
    batch_size = 32
    num_frames = 800

    model = ConvTasNet()

    def data_gen():
        tensor = torch.rand(batch_size, 1, num_frames)
        return dict(input=tensor)

    trace_and_compare(model, data_gen, need_meta=True, need_concrete=False)


def test_deepspeech():
    n_batch = 2
    n_feature = 1
    n_channel = 1
    n_class = 40
    n_time = 32

    model = DeepSpeech(n_feature=n_feature, n_class=n_class)

    def data_gen():
        x = torch.rand(n_batch, n_channel, n_time, n_feature)
        return dict(x=x)

    trace_and_compare(model, data_gen, need_meta=False, need_concrete=False)


if __name__ == '__main__':
    TEST_LIST = [
        test_wave2letter_waveform,
        test_wave2letter_mfcc,
        test_melresnet_waveform,
        test_upsample_network_waveform,
        test_wavernn_waveform,
        test_convtasnet_config,
        test_deepspeech,
    ]

    for test_fn in TEST_LIST:
        test_fn()
