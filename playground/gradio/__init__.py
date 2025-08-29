# lm_gradio_interface.py
__all__ = ["LMGradioInterface"]

import time
import numpy as np
import gradio as gr
from threading import Lock

from live_mind.controller.abc import BaseStreamController
from audio.stt import SileroSTT

CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto; }
"""

class LMGradioInterface:
    """
    Gradio UI:
      - Chatbot + streaming inference box (your original behavior)
      - Live microphone (continuous) -> Silero STT -> appends transcript to msg textbox

    Notes:
      - We buffer ~1.2s of mic audio before transcribing for better quality.
      - Simple, robust dedup avoidance: only transcribe when we flush a buffer.
    """
    def __init__(
        self,
        lm_controller: BaseStreamController,
        base_controller: BaseStreamController,
        stt_device: str = "cpu"
    ):
        self.lm_controller = lm_controller
        self.base_controller = base_controller

        # Streaming state
        self.lock = Lock()
        self.use_lm = True
        self.infer_msg = ""
        self.input_msg = ""

        # --- STT state ---
        self.stt = SileroSTT(device=stt_device)
        self._stt_buf = bytearray()
        self._stt_sr = 16000
        self._stt_min_secs = 1.2   # buffer ~1.2s before transcribing
        self._last_transcript_ts = 0.0

        self.on_mount()

    # ---------- Helpers ----------
    @staticmethod
    def _ensure_int16_mono(data: np.ndarray) -> np.ndarray:
        """
        Convert incoming audio buffer (float or int, mono or stereo) to int16 mono.
        data shape could be (n,) or (n, channels)
        """
        if data is None or data.size == 0:
            return np.zeros((0,), dtype=np.int16)

        # If stereo, average to mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Convert to int16
        if data.dtype == np.int16:
            pcm = data
        else:
            # Assume float in [-1, 1]
            data = np.clip(data.astype(np.float32), -1.0, 1.0)
            pcm = (data * 32767.0).astype(np.int16)

        return pcm

    def _enough_audio(self, sample_rate: int) -> bool:
        # 2 bytes per sample (int16 mono)
        needed = int(self._stt_min_secs * sample_rate * 2)
        return len(self._stt_buf) >= needed

    # ---------- UI ----------
    def on_mount(self):
        title = "Live Mind Chat Interface"
        with gr.Blocks(css=CSS, title=title) as demo:
            chatbot = gr.Chatbot(elem_id="chatbot")
            infer_box = gr.Textbox("", interactive=False, label="Actions", max_lines=15)
            msg = gr.Textbox(placeholder="Type here...", label="Input")
            clear = gr.Button("Clear")
            use_lm = gr.Checkbox(label="Use LM framework", value=self.use_lm)
            show_infer = gr.Checkbox(label="Show inference", value=True)

            # NEW: live mic (continuous)
            mic = gr.Audio(
                sources="microphone",
                streaming=True,
                label="🎤 Mic (continuous STT → appends into Input)",
            )

            # -------- original handlers --------
            def clear_input(msg_text):
                # Move the visible textbox content into our rolling input_msg
                self.input_msg += msg_text
                return ""

            def update_input():
                # After submit, add a newline into rolling buffer
                self.input_msg += "\n"

            def action_submit(use_lm_val, chatbot_state):
                chatbot_state += [[self.input_msg, ""]]
                with self.lock:
                    controller = self.lm_controller if use_lm_val else self.base_controller
                    for response in controller.iter_call(self.input_msg, stream_end=True):
                        for text in response:
                            chatbot_state[-1][1] += text
                            yield chatbot_state
                    yield chatbot_state

            def action_change(text, use_lm_val):
                if not use_lm_val:
                    return
                with self.lock:
                    text = self.input_msg + text
                    for response in self.lm_controller.iter_call(text):
                        if self.infer_msg != "":
                            self.infer_msg += "\n"
                        for s in response:
                            self.infer_msg += s
                            yield self.infer_msg
                    yield self.infer_msg

            def action_clear():
                self.infer_msg = ""
                self.input_msg = ""
                self.base_controller.reset()
                self.lm_controller.reset()
                # Reset STT buffer too
                self._stt_buf.clear()
                return None, self.infer_msg

            def change_visibility(show):
                return gr.Textbox(visible=bool(show))

            # -------- NEW: mic → STT → append into msg --------
            def stt_append_and_infer(audio, current_text, use_lm_val):
                """
                Streamed handler for mic:
                1) Append transcribed chunk to msg
                2) If LM is enabled, run streaming inference on (self.input_msg + msg)
                    and yield incremental updates to infer_box.
                Yields: (updated_msg, updated_infer_box_text)
                """
                try:
                    # --- 1) same buffering/transcribe logic you already have ---
                    if audio is None:
                        # Nothing to do; keep infer_box as-is
                        yield current_text, self.infer_msg
                        return

                    # Accept dict {"sampling_rate": int, "data": list/ndarray} or (sr, data)
                    if isinstance(audio, dict):
                        sr = int(audio.get("sampling_rate", 16000))
                        data = np.array(audio.get("data", []))
                    else:
                        sr, data = audio
                        sr = int(sr)
                        data = np.array(data)

                    self._stt_sr = sr
                    pcm = self._ensure_int16_mono(data)
                    if pcm.size > 0:
                        self._stt_buf += pcm.tobytes()

                    # Only transcribe when enough audio is buffered
                    if not self._enough_audio(sr):
                        # Yield current state so UI stays responsive
                        yield current_text, self.infer_msg
                        return

                    # Flush buffer window and run STT
                    raw = bytes(self._stt_buf)
                    self._stt_buf.clear()
                    chunk_text = self.stt.run_stt(raw_bytes=raw, sample_rate=sr)

                    # Append into the visible textbox
                    if chunk_text:
                        base = current_text or ""
                        sep = "" if base.endswith((" ", "\n", "")) else " "
                        new_msg = base + sep + chunk_text
                    else:
                        new_msg = current_text

                    # First, immediately show the updated msg (no inference yet)
                    yield new_msg, self.infer_msg

                    # --- 2) run streaming inference like action_change does ---
                    if not use_lm_val:
                        return

                    with self.lock:
                        text_for_infer = self.input_msg + new_msg
                        for response in self.lm_controller.iter_call(text_for_infer):
                            if self.infer_msg != "":
                                self.infer_msg += "\n"
                            for s in response:
                                self.infer_msg += s
                                # Stream both: keep msg fixed, update infer_box incrementally
                                yield new_msg, self.infer_msg

                        # Final yield to flush any remaining UI updates
                        yield new_msg, self.infer_msg

                except Exception:
                    # Be resilient; don't crash the stream on one bad chunk
                    yield current_text, self.infer_msg


            # Wire up events
            msg.submit(clear_input, [msg], msg, queue=True)\
               .then(action_submit, [use_lm, chatbot], [chatbot], queue=True)\
               .then(update_input, [], queue=True)

            msg.change(action_change, [msg, use_lm], infer_box, show_progress=False, queue=True)
            clear.click(action_clear, [], [chatbot, infer_box], queue=True)
            show_infer.change(change_visibility, show_infer, infer_box, show_progress=False)

            # Mic streams into stt_append which updates the msg textbox
            mic.stream(
                stt_append_and_infer,              # new handler below
                [mic, msg, use_lm],               # inputs
                [msg, infer_box],                 # outputs: update both msg and infer_box
                queue=True,
                show_progress=False
            )
        self.demo = demo

    def run(self):
        self.demo.launch()
