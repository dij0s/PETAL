import os

from langchain_ollama import ChatOllama

class ModelProvider():
    """
    The following class implements a factory pattern for model instance creation.
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError("ModelProvider cannot be instantiated. Use class methods only.")

    @staticmethod
    def from_env_variable(env_variable: str, temperature: float, defaults: str, **kwargs) -> ChatOllama:
        """
        Creates a ChatOllama model instance using parameters from an environment variable.

        Args:s
            env_variable (str): The name of the environment variable containing the model name.
            temperature (float): The temperature parameter for the model.
            defaults (str): Default model name if the environment variable is not set.

        Returns:
            ChatOllama: A ChatOllama model instance.
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
