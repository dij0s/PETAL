import os

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackManager
from langchain_ollama import ChatOllama

from typing import Any, Optional

class MyCustomCallback(AsyncCallbackHandler):
    async def on_chat_model_start(self, serialized: dict[str, Any], messages, *, run_id, parent_run_id = None, tags: Optional[list[str]] = None, metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        return await super().on_chat_model_start(serialized, messages, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)

    async def on_llm_end(self, response, *, run_id, parent_run_id = None, tags: Optional[list[str]] = None, **kwargs: Any) -> None:
        return await super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, tags=tags, **kwargs)

class ChatOllamaWrapper(ChatOllama):
    """
    The following class wraps the ChatOllama implementation.
    """

    def __init__(self, model: str, temperature: float, **kwargs):
        handlers = kwargs.get("callbacks", []) + [MyCustomCallback()]
        manager: BaseCallbackManager = BaseCallbackManager(handlers=handlers, inheritable_handlers=handlers)
        super().__init__(model=model, temperature=temperature, **kwargs, callbacks=manager)

    async def ainvoke(
        self,
        input,
        config=None,
        *,
        stop=None,
        **kwargs
    ):
        return await super().ainvoke(input=input, config=config, stop=stop, **kwargs)

class ModelProvider():
    """
    The following class implements a factory pattern for model instance creation.
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError("ModelProvider cannot be instantiated. Use class methods only.")

    @staticmethod
    def from_env_variable(env_variable: str, temperature: float, defaults: str, **kwargs) -> ChatOllamaWrapper:
        """
        Creates a ChatOllamaWrapper model instance using parameters from an environment variable.

        Args:s
            env_variable (str): The name of the environment variable containing the model name.
            temperature (float): The temperature parameter for the model.
            defaults (str): Default model name if the environment variable is not set.

        Returns:
            ChatOllamaWrapper: A wrapper of the ChatOllama model instance.
        """
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "")
        MODEL = os.getenv(env_variable, defaults)

        if OLLAMA_HOST == "":
            # default to local
            # ollama installation
            # if host is not set
            return ChatOllamaWrapper(model=MODEL, temperature=temperature, **kwargs)
        else:
            return ChatOllamaWrapper(model=MODEL, temperature=temperature, base_url=OLLAMA_HOST, **kwargs)
