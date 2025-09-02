import os

from langchain.chat_models.base import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

class ModelProvider():
    """
    The following class implements a factory pattern for model instance creation.
    """

    @staticmethod
    def from_ollama(env_variable: str, temperature: float, defaults: str, **kwargs) -> BaseChatModel:
        """
        Creates a ChatOllama model instance using parameters from an environment variable.

        Args:s
            env_variable (str): The name of the environment variable containing the model name.
            temperature (float): The temperature parameter for the model.
            defaults (str): Default model name if the environment variable is not set.

        Returns:
            ChatOllama: An instance of the ChatOllama model.
        """
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "")
        MODEL = os.getenv(env_variable, defaults)

        if OLLAMA_HOST == "":
            # default to local
            # ollama installation
            # if host is not set
            return ChatOllama(model=MODEL, temperature=temperature, **kwargs)
        else:
            return ChatOllama(model=MODEL, temperature=temperature, base_url=OLLAMA_HOST, **kwargs)

    @staticmethod
    def from_hf(env_variable: str, **kwargs) -> BaseChatModel:
        """
        Creates a ChatHuggingFace model instance using parameters from an environment variable.

        Args:s
            env_variable (str): The name of the environment variable containing the model name.
            temperature (float): The temperature parameter for the model.

        Returns:
            ChatHuggingFace: An instance of the ChatHuggingFace model.
        """
        MODEL = os.getenv(env_variable, "")
        if MODEL == "":
            raise ValueError("Model name not found in environment variable")

        llm = HuggingFaceEndpoint(
            repo_id=MODEL,
            **kwargs
        )
        return ChatHuggingFace(llm=llm)
