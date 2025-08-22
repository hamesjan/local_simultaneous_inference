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