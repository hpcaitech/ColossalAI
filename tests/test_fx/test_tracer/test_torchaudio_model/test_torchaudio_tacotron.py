import torch
from torchaudio.models import Tacotron2
from torchaudio_utils import trace_and_compare
import pytest


def _get_tacotron2_model(n_mels, decoder_max_step=2000, gate_threshold=0.5):
    return Tacotron2(
        mask_padding=False,
        n_mels=n_mels,
        n_symbol=20,
        n_frames_per_step=1,
        symbol_embedding_dim=32,
        encoder_embedding_dim=32,
        encoder_n_convolution=3,
        encoder_kernel_size=5,
        decoder_rnn_dim=32,
        decoder_max_step=decoder_max_step,
        decoder_dropout=0.1,
        decoder_early_stopping=True,
        attention_rnn_dim=32,
        attention_hidden_dim=32,
        attention_location_n_filter=32,
        attention_location_kernel_size=31,
        attention_dropout=0.1,
        prenet_dim=32,
        postnet_n_convolution=5,
        postnet_kernel_size=5,
        postnet_embedding_dim=512,
        gate_threshold=gate_threshold,
    )


@pytest.mark.skip("Tracing failed")
def test_tacotron_model():
    n_mels = 80
    n_batch = 3
    max_mel_specgram_length = 300
    max_text_length = 100

    model = _get_tacotron2_model(n_mels)

    def data_gen():
        text = torch.randint(0, 148, (n_batch, max_text_length))
        text_lengths = max_text_length * torch.ones((n_batch,))
        mel_specgram = torch.rand(n_batch, n_mels, max_mel_specgram_length)
        mel_specgram_lengths = max_mel_specgram_length * torch.ones((n_batch,))
        return dict(tokens=text,
                    token_lengths=text_lengths,
                    mel_specgram=mel_specgram,
                    mel_specgram_lengths=mel_specgram_lengths)

    trace_and_compare(model, data_gen, need_meta=True, need_concrete=False)


if __name__ == "__main__":
    test_tacotron_model()
