from functools import partial

import torch
import torchaudio.models as tm

from ..registry import ModelAttribute, model_zoo

INPUT_DIM = 80
IN_FEATURES = 16
N_TIME = 20
KERNEL_SIZE = 5
HOP_LENGTH = 20
N_CLASSES = 10
N_FREQ = 16
N_MELS = 80


def conformer_data_gen_fn():
    lengths = torch.randint(1, 400, (4,))
    input = torch.rand(4, int(lengths.max()), INPUT_DIM)
    return dict(input=input, lengths=lengths)


transformer_output_transform_fn = lambda outputs: dict(frames=outputs[0], lengths=outputs[1])

model_zoo.register(
    name="torchaudio_conformer",
    model_fn=lambda: tm.Conformer(
        input_dim=INPUT_DIM, num_heads=4, ffn_dim=128, num_layers=4, depthwise_conv_kernel_size=31
    ),
    data_gen_fn=conformer_data_gen_fn,
    output_transform_fn=transformer_output_transform_fn,
)

single_output_transform_fn = lambda output: dict(output=output)

model_zoo.register(
    name="torchaudio_convtasnet",
    model_fn=tm.ConvTasNet,
    data_gen_fn=lambda: dict(input=torch.rand(4, 1, 8)),
    output_transform_fn=single_output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)

model_zoo.register(
    name="torchaudio_deepspeech",
    model_fn=lambda: tm.DeepSpeech(IN_FEATURES, n_hidden=128, n_class=4),
    data_gen_fn=lambda: dict(x=torch.rand(4, 1, 10, IN_FEATURES)),
    output_transform_fn=single_output_transform_fn,
)


def emformer_data_gen_fn():
    input = torch.rand(4, 400, IN_FEATURES)
    lengths = torch.randint(1, 200, (4,))
    return dict(input=input, lengths=lengths)


model_zoo.register(
    name="torchaudio_emformer",
    model_fn=lambda: tm.Emformer(input_dim=IN_FEATURES, num_heads=4, ffn_dim=128, num_layers=4, segment_length=4),
    data_gen_fn=emformer_data_gen_fn,
    output_transform_fn=transformer_output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)

model_zoo.register(
    name="torchaudio_wav2letter_waveform",
    model_fn=lambda: tm.Wav2Letter(input_type="waveform", num_features=40),
    data_gen_fn=lambda: dict(x=torch.rand(4, 40, 400)),
    output_transform_fn=single_output_transform_fn,
)

model_zoo.register(
    name="torchaudio_wav2letter_mfcc",
    model_fn=lambda: tm.Wav2Letter(input_type="mfcc", num_features=40),
    data_gen_fn=lambda: dict(x=torch.rand(4, 40, 400)),
    output_transform_fn=single_output_transform_fn,
)


def wavernn_data_gen_fn():
    waveform = torch.rand(4, 1, (N_TIME - KERNEL_SIZE + 1) * HOP_LENGTH)
    specgram = torch.rand(4, 1, N_FREQ, N_TIME)
    return dict(waveform=waveform, specgram=specgram)


model_zoo.register(
    name="torchaudio_wavernn",
    model_fn=lambda: tm.WaveRNN(
        upsample_scales=[2, 2, 5],
        n_classes=N_CLASSES,
        hop_length=HOP_LENGTH,
        kernel_size=KERNEL_SIZE,
        n_freq=N_FREQ,
        n_res_block=2,
        n_rnn=64,
        n_fc=64,
        n_hidden=16,
        n_output=16,
    ),
    data_gen_fn=wavernn_data_gen_fn,
    output_transform_fn=single_output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)


def tacotron_data_gen_fn():
    n_batch = 4
    max_text_length = 100
    max_mel_specgram_length = 300
    tokens = torch.randint(0, 148, (n_batch, max_text_length))
    token_lengths = max_text_length * torch.ones((n_batch,))
    mel_specgram = torch.rand(n_batch, N_MELS, max_mel_specgram_length)
    mel_specgram_lengths = max_mel_specgram_length * torch.ones((n_batch,))
    return dict(
        tokens=tokens, token_lengths=token_lengths, mel_specgram=mel_specgram, mel_specgram_lengths=mel_specgram_lengths
    )


model_zoo.register(
    name="torchaudio_tacotron",
    model_fn=lambda: tm.Tacotron2(n_mels=N_MELS),
    data_gen_fn=tacotron_data_gen_fn,
    output_transform_fn=lambda outputs: dict(summed_output=sum(x.sum() for x in outputs)),
    model_attribute=ModelAttribute(has_control_flow=True),
)


def wav2vec_data_gen_fn():
    batch_size, num_frames = 4, 400
    waveforms = torch.randn(batch_size, num_frames)
    lengths = torch.randint(0, num_frames, (batch_size,))
    return dict(waveforms=waveforms, lengths=lengths)


model_zoo.register(
    name="torchaudio_wav2vec2_base",
    model_fn=partial(tm.wav2vec2_base, encoder_layer_drop=0.0),
    data_gen_fn=wav2vec_data_gen_fn,
    output_transform_fn=transformer_output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)

model_zoo.register(
    name="torchaudio_hubert_base",
    model_fn=tm.hubert_base,
    data_gen_fn=wav2vec_data_gen_fn,
    output_transform_fn=transformer_output_transform_fn,
    model_attribute=ModelAttribute(has_control_flow=True),
)
