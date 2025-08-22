""" Configuration file """
from abc import ABC, abstractmethod
from live_mind.models.ollama_adapter import OllamaAdapter

# the dataset path should contain the `.parquet` files
# path to the MMLU-PRO dataset
MMLU_PRO_PATH = "/home/hendrix/LiveMind/MMLU-Pro"
# path to the MMLU dataset
MMLU_PATH = "/home/hendrix/LiveMind/mmlu"
# config the model path if you use 'get_model_vllm_example'
# the model path should contain a `config.json` file

LLAMA_MODELS = ["llama-3.2:1b"]

def get_model(name: str):
    return get_model_ollama_example(name)


def get_model_ollama_example(name: str) -> 'BaseModel':
    """
    Minimal Ollama adapter implementing the same BaseModel interface as the OpenAI/Anthropic examples.
    Expects `message` in chat_complete to be a list of {"role": "...", "content": "..."}.
    """
    import requests
    assert name in OLLAMA_MODELS, f"Model {name} is not supported."

    OLLAMA_URL = "http://localhost:11434/api/chat"
    TIMEOUT = 120  # seconds

    class Model(BaseModel):
        def chat_complete(self, message):
            """
            message: list[dict], e.g. [{"role":"system","content":"..."},
                                       {"role":"user","content":"..."}]
            Returns: response text (string)
            """
            # Basic payload: forward messages as-is (Ollama uses the same role keys)
            payload = {
                "model": name,
                "messages": message,
                # you can add other options here if desired, e.g. temperature, stop, etc.
                "temperature": 0.0,
            }

            try:
                resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
                resp.raise_for_status()
                j = resp.json()
            except Exception as e:
                # Keep failure mode similar to other adapters: return empty string on failure
                # Optionally you can log/print e for debugging
                print(f"[OllamaAdapter] request error: {e}")
                return ""

            # Try common shapes for Ollama responses
            # 1) {"choices":[{"message":{"content":"..."}}]}
            try:
                return j["choices"][0]["message"]["content"] or ""
            except Exception:
                pass

            # 2) {"choices":[{"content":"..."}]} or {"choices":[{"text":"..."}]}
            try:
                choice0 = j.get("choices", [{}])[0]
                if "content" in choice0 and isinstance(choice0["content"], str):
                    return choice0["content"]
                if "text" in choice0 and isinstance(choice0["text"], str):
                    return choice0["text"]
            except Exception:
                pass

            # 3) fallback: try top-level 'output' or just stringify JSON
            try:
                if "output" in j and isinstance(j["output"], str):
                    return j["output"]
            except Exception:
                pass

            # nothing matched: return stringified json (or empty)
            try:
                return str(j)
            except Exception:
                return ""

    return Model()


class BaseModel(ABC):
    """ Base model class """
    @abstractmethod
    def chat_complete(self, message: list[dict[str, str]]) -> str:
        """ The message should be a list of dictionaries with the following keys:
        - role: "system", "user", or "assistant" (all these roles should be supported)
        - content: the content of the message
        """
        pass

