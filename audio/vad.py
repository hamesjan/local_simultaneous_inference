# audio/vad_stream.py
import time
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

class VADState:
    """
    Feed PCM16 mono chunks to .process_chunk(raw_bytes) repeatedly.
    When silence persists for `speech_pause_sec` after speech, it returns a finalized utterance.
    """
    def __init__(self,
                 sample_rate=16000,
                 chunk_sec=0.5,
                 speech_pause_sec=1.0,
                 vad_device=None):
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.speech_pause_sec = speech_pause_sec
        self.vad_model = vad_device or load_silero_vad()
        self._speech_active = False
        self._speech_buf = bytearray()
        self._last_speech_t = 0.0

    def process_chunk(self, raw_bytes: bytes):
        """
        Returns:
          (partial_text_hint, final_bytes) where:
            - partial_text_hint is always "" in this minimal impl (hook for live partials)
            - final_bytes is b"" most of the time; when an utterance ends, it contains
              the full PCM16 bytes for that utterance and we reset the buffer.
        """
        if not raw_bytes:
            # check timeout with no new audio
            return "", self._try_finalize()

        # VAD on this chunk
        f32 = (np.frombuffer(raw_bytes, np.int16).astype(np.float32) / 32768.0)
        speech_ts = get_speech_timestamps(
            f32,
            self.vad_model,
            sampling_rate=self.sample_rate,
            return_seconds=False
        )

        now = time.time()

        if len(speech_ts) > 0:
            # speech present
            self._speech_buf.extend(raw_bytes)
            self._speech_active = True
            self._last_speech_t = now
            return "", b""

        # no speech in this chunk
        if self._speech_active and (now - self._last_speech_t) > self.speech_pause_sec:
            # End of utterance
            return "", self._finalize()

        return "", b""

    def _try_finalize(self):
        if self._speech_active and (time.time() - self._last_speech_t) > self.speech_pause_sec:
            return self._finalize()
        return b""

    def _finalize(self):
        out = bytes(self._speech_buf)
        self._speech_buf.clear()
        self._speech_active = False
        return out
