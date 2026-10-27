"""Application settings, read from environment variables and the .env file."""

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from claim_extractor.llm.base import LLMClient
from claim_extractor.llm.mistral import MistralClient
from claim_extractor.llm.ollama import OllamaClient


class Settings(BaseSettings):
    """Each field maps to the environment variable of the same name, in upper case.

    Real environment variables take precedence over the .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,  # `MISTRAL_API_KEY=` in .env means "not set"
        extra="ignore",  # other variables in .env are not our concern
    )

    llm_provider: Literal["ollama", "mistral"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    mistral_api_key: SecretStr | None = None  # SecretStr: shown as '**********' if printed
    mistral_model: str = "mistral-small-latest"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


def build_client(settings: Settings) -> LLMClient:
    """Create the LLM client selected by LLM_PROVIDER."""
    if settings.llm_provider == "mistral":
        if settings.mistral_api_key is None:
            raise ValueError(
                "MISTRAL_API_KEY is not set: copy .env.example to .env and fill it in."
            )
        return MistralClient(
            settings.mistral_model, api_key=settings.mistral_api_key.get_secret_value()
        )
    return OllamaClient(settings.ollama_model, base_url=settings.ollama_base_url)
