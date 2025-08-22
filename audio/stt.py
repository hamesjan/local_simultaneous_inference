# audio/stt.py
import torch
import numpy as np

class SileroSTT:
    def __init__(self, device="cpu", language="en"):
        print("-> Loading Silero STT model...")
        model_stt, decoder, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_stt',
            language=language,
            device=device
        )
        # utils: read_batch, split_into_batches, read_audio, prepare_model_input
        (_, _, _, prepare_model_input) = utils

        self.model = model_stt
        self.decoder = decoder
        self.prepare_model_input = prepare_model_input
        self.device = device

    def run_stt_bytes(self, raw_bytes: bytes, sample_rate=16000) -> str:
        """
        Run STT directly on raw PCM16 mono bytes at `sample_rate` (default 16k).
        Returns decoded text.
        """
        # bytes -> int16 -> float32 in [-1, 1]
        pcm16 = np.frombuffer(raw_bytes, dtype=np.int16)
        if pcm16.size == 0:
            return ""
        audio_f32 = (pcm16.astype(np.float32) / 32768.0)

        # Pack as list[tensor] the way Silero utils expect
        # (prepare_model_input handles padding / batching)
        audio_t = torch.from_numpy(audio_f32)
        input_data = self.prepare_model_input([audio_t], device=self.device, sample_rate=sample_rate)

        with torch.inference_mode():
            logits = self.model(input_data)
        text = self.decoder(logits[0]).strip()
        return text
