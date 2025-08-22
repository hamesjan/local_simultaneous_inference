__all__ = ["LMGradioInterface"]

import numpy as np
import gradio as gr
from threading import Lock
from live_mind.controller.abc import BaseStreamController

# === your audio backends ===
from audio.stt import SileroSTT              # must expose .run_stt_bytes(raw_bytes, sample_rate)
from audio.vad_stream import VADState        # must expose .process_chunk(raw_bytes)-> ("", final_bytes) and ._try_finalize()

CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto; }
"""

class LMGradioInterface:
    """
    Voice-first interface:
      - Mic streams float32 [-1,1] chunks from the browser
      - VAD accumulates speech and finalizes on silence
      - STT runs on finalized bytes
      - Controller streams reply into the chatbot
    """
    def __init__(
        self,
        lm_controller: BaseStreamController,
        base_controller: BaseStreamController,
        sample_rate: int = 16000,
        speech_pause_sec: float = 0.9,
    ):
        self.lm_controller = lm_controller
        self.base_controller = base_controller
        self.lock = Lock()

        self.sample_rate = sample_rate
        self.vad = VADState(sample_rate=sample_rate, speech_pause_sec=speech_pause_sec)
        self.stt = SileroSTT(device="cpu")  # set to "cuda" if you have a GPU

        self.use_lm_default = True
        self.on_mount()

    def on_mount(self):
        title = "Live Mind – Voice Chat"
        with gr.Blocks(css=CSS, title=title) as demo:
            chatbot = gr.Chatbot(elem_id="chatbot")
            infer_box = gr.Textbox("", interactive=False, label="Actions", max_lines=15, value="")
            use_lm = gr.Checkbox(label="Use LM framework", value=self.use_lm_default)
            show_infer = gr.Checkbox(label="Show inference", value=True)
            clear = gr.Button("Clear")

            # Always-on mic (streams small waveform chunks)
            mic = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",                 # np.float32 waveform in [-1, 1]
                sample_rate=self.sample_rate,
                label="🎙️ Mic is live while this tab is open",
            )

            # Keep a rolling text state if you want to show partials later
            transcript_state = gr.State("")

            def on_mic_chunk(audio_np, use_lm_val, transcript_so_far):
                """
                Called on each incoming mic chunk.
                - Feed chunk to VAD
                - When VAD returns a finalized utterance, run STT
                - Stream reply via controller
                """
                # If no new audio, check if silence ended an utterance
                if audio_np is None:
                    final_bytes = self.vad._try_finalize()
                else:
                    # Gradio gives float32 [-1,1]; convert to PCM16 bytes for VAD/STT
                    if audio_np.dtype != np.float32:
                        audio_np = audio_np.astype(np.float32)
                    pcm16 = (audio_np * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
                    _, final_bytes = self.vad.process_chunk(pcm16)

                # Nothing finalized yet → no UI change
                if not final_bytes:
                    return gr.update(), transcript_so_far, gr.update(visible=show_infer.value)

                # STT on the finalized utterance
                text = self.stt.run_stt_bytes(final_bytes, sample_rate=self.sample_rate).strip()

                if not text:
                    return gr.update(), transcript_so_far, gr.update(visible=show_infer.value)

                # Append user bubble and stream assistant reply
                chat = chatbot + [[text, ""]]
                with self.lock:
                    controller = self.lm_controller if use_lm_val else self.base_controller
                    for response in controller.iter_call(text, stream_end=True):
                        for token in response:
                            chat[-1][1] += token
                            # yield incremental updates to the UI
                            yield chat, "", gr.update(visible=show_infer.value)

                # Final UI state for this utterance
                yield chat, "", gr.update(visible=show_infer.value)

            def action_clear():
                self.lm_controller.reset()
                self.base_controller.reset()
                # Also clear any partial VAD state
                self.vad._speech_buf.clear()
                return None, "", ""

            def change_visibility(v):
                return gr.Textbox(visible=bool(v))

            # Wire events
            mic.stream(
                on_mic_chunk,
                inputs=[mic, use_lm, transcript_state],
                outputs=[chatbot, transcript_state, infer_box],
                queue=True
            )
            clear.click(action_clear, [], [chatbot, transcript_state, infer_box], queue=True)
            show_infer.change(change_visibility, show_infer, infer_box, show_progress=False)

        self.demo = demo

    def run(self):
        self.demo.queue().launch()
