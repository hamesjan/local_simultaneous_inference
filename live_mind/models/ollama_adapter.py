# file: live_mind/models/ollama_adapter.py
import requests
import json
from typing import List, Dict, Iterable

class OllamaAdapter:
    """
    Minimal adapter to call Ollama REST /api/chat.
    Methods below are minimal; adapt names to LiveMind's BaseModel interface if needed.
    """

    def __init__(self, model_name: str = "llama3.2:1b", url: str = "http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.url = url

    def chat(self, messages: List[Dict[str,str]], **kwargs) -> Dict:
        """
        messages: list of {"role":"user"|"system"|"assistant", "content": "..."}
        returns: full JSON response from Ollama
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
        }
        payload.update(kwargs or {})
        r = requests.post(self.url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Convenience: one-shot prompt -> text"""
        messages = [{"role": "user", "content": prompt}]
        j = self.chat(messages, **kwargs)
        # Ollama returns `choices` / content depending on version; adjust accordingly:
        # Try to extract assistant content if present.
        try:
            # common shape: {"choices":[{"message":{"content":"..."}}, ...]}
            return j["choices"][0]["message"]["content"]
        except Exception:
            # fallback: return raw json
            return json.dumps(j)

    def stream_chat(self, messages: List[Dict[str,str]], chunk_handler):
        """
        Minimal chunked/streaming example — if you want to stream output into LiveMind UI.
        chunk_handler should be callable(chunk_str)
        Ollama supports chunked responses; here we do a simple requests iter_lines approach.
        """
        payload = {"model": self.model_name, "messages": messages}
        with requests.post(self.url, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # server may send JSON chunks or text lines; pass to handler
                chunk_handler(line)
