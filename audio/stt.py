# audio/stt.py
import os
import wave
import torch


class SileroSTT:
    """
    Lightweight wrapper around Silero STT loaded from torch.hub.

    Usage:
        stt = SileroSTT(device="cpu")  # or "cuda" if available
        text = stt.run_stt(raw_pcm_bytes, sample_rate=16000)
    """
    def __init__(self, device: str = "cpu", language: str = "en"):
        print("-> Loading Silero STT model...")
        # This will download/cache the model the first time it runs.
        # Repo: https://github.com/snakers4/silero-models
        model_stt, decoder, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_stt",
            language=language,
            device=device
        )
        (read_batch, split_into_batches, read_audio, prepare_model_input) = utils
        self.model = model_stt
        self.decoder = decoder
        self.read_audio = read_audio
        self.prepare_model_input = prepare_model_input
        self.device = device

    def run_stt(self, raw_bytes: bytes, sample_rate: int = 16000, temp_wav: str = "temp_stt.wav") -> str:
        """
        Writes raw PCM (int16 mono) bytes to a temp WAV, runs Silero STT, returns text.
        - raw_bytes: PCM 16-bit mono bytes
        - sample_rate: e.g., 16000, 44100. Silero utils will handle resampling internally.
        """
        # Write a small temporary WAV file
        with wave.open(temp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(int(sample_rate))
            wf.writeframes(raw_bytes)

        try:
            audio_tensor = self.read_audio(temp_wav)
            input_data = self.prepare_model_input([audio_tensor], device=self.device)
            output = self.model(input_data)
            text = self.decoder(output[0])
        finally:
            try:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            except Exception:
                # Non-fatal if cleanup fails
                pass

        return (text or "").strip()
