# playground/ollama_session.py
import os, json, requests
from typing import Generator, List, Dict, Any
from live_mind.abc import BaseStreamModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))

class Session(BaseStreamModel):
    """
    Ollama-backed Session. Implements:
      - chat_complete(messages) -> str
      - stream(messages) -> Generator[str, None, None]
    `messages` must be a list[{"role": "system"|"user"|"assistant", "content": str}]
    """
    def __init__(self, model: str, temperature: float = 0.0, top_p: float = 0.9, max_gen_len: int = 8192):
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_gen_len = int(max_gen_len)

    def _payload(self, messages: List[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_gen_len,
            },
        }

    def chat_complete(self, message: List[Dict[str, str]]) -> str:
        # Non-streaming
        try:
            resp = requests.post(OLLAMA_URL, json=self._payload(message, stream=False), timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            j = resp.json()
        except Exception as e:
            print(f"[Ollama.chat_complete] {e}")
            return ""
        # Shapes:
        # /api/chat (non-stream): {"message":{"role":"assistant","content":"..."}, "done": true, ...}
        if isinstance(j, dict):
            msg = j.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content
            # /api/generate fallback: {"response":"..."}
            if isinstance(j.get("response"), str):
                return j["response"]
        return ""

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

