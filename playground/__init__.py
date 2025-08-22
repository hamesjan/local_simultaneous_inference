from live_mind.abc import BaseStreamModel
from collections.abc import Generator
from . import gradio
from . import ollama_session   # NEW

__all__ = ["gradio"]

LLAMA_MODELS = ["llama3.2:1b"]
SUPPORTED_MODELS = LLAMA_MODELS

def get_stream_model(name: str) -> BaseStreamModel:
    assert name in SUPPORTED_MODELS, f"Model {name} is not supported."
    if name in LLAMA_MODELS:
        return ollama_session.Session(name, temperature=0.0)
    raise ValueError("Unreachable")

def _get_stream_ollama_model(name: str) -> BaseStreamModel:
    """
    Adapter for Ollama REST API. Implements BaseStreamModel.chat_complete(message)
    and BaseStreamModel.stream(message) -> Generator[str, None, None]
    where `message` is a list[dict{"role":..., "content":...}].
    """
    class OllamaModel(BaseStreamModel):
        def chat_complete(self, message: list[dict[str, str]]) -> str:
            payload = {"model": name, "messages": message, "temperature": 0.0}
            try:
                resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
                resp.raise_for_status()
                j = resp.json()
            except Exception as e:
                print(f"[OllamaModel.chat_complete] request error: {e}")
                return ""

            # Typical shapes: {"choices":[{"message":{"content":"..."}}]} or {"choices":[{"content":"..."}]}
            try:
                return j["choices"][0]["message"]["content"] or ""
            except Exception:
                pass
            try:
                c0 = j.get("choices", [{}])[0]
                if isinstance(c0.get("content"), str):
                    return c0["content"]
                if isinstance(c0.get("text"), str):
                    return c0["text"]
            except Exception:
                pass
            if isinstance(j.get("output"), str):
                return j["output"]
            return str(j)

        def stream(self, message: list[dict[str, str]]):
            import json, requests
            payload = {
                "model": self.model,
                "messages": message,
                "stream": True,
                "options": {"temperature": self.temperature, "top_p": self.top_p, "num_predict": self.max_gen_len},
            }
            try:
                with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as r:
                    r.raise_for_status()
                    # Keep raw as bytes; strip/startswith using bytes; decode AFTER
                    for raw in r.iter_lines(chunk_size=8192, decode_unicode=False):
                        if not raw:
                            continue
                        if isinstance(raw, (bytes, bytearray)):
                            bline = raw.strip()
                            # tolerate SSE-style "data:" prefixes at the BYTES level
                            if bline.startswith(b"data:"):
                                bline = bline[5:].strip()
                            if not bline:
                                continue
                            line = bline.decode("utf-8", errors="ignore")
                        else:
                            # fallback, shouldn't happen but safe
                            line = str(raw).strip()
                            if line.startswith("data:"):
                                line = line[5:].strip()
                            if not line:
                                continue

                        # Expect NDJSON per chunk
                        try:
                            chunk = json.loads(line)
                        except Exception:
                            # uncomment to debug odd frames:
                            # print("nonjson:", repr(line))
                            continue

                        # /api/chat streaming
                        msg = chunk.get("message")
                        if isinstance(msg, dict):
                            text = msg.get("content")
                            if isinstance(text, str) and text:
                                yield text
                        # /api/generate streaming
                        elif isinstance(chunk.get("response"), str):
                            yield chunk["response"]

                        if chunk.get("done"):
                            break
            except Exception as e:
                print(f"[Ollama.stream] {e.__class__.__name__}: {e}")
                return

               
    return OllamaModel()
