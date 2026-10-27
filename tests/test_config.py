from pathlib import Path

import pytest
from pydantic import ValidationError

from claim_extractor.config import Settings, build_client
from claim_extractor.llm.mistral import MistralClient
from claim_extractor.llm.ollama import OllamaClient


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from the developer's own environment variables."""
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


def write_env_file(tmp_path: Path, content: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(content, encoding="utf-8")
    return env_file


def test_defaults_without_any_configuration() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "ollama"
    assert settings.mistral_api_key is None


def test_reads_the_dotenv_file(tmp_path: Path) -> None:
    env_file = write_env_file(tmp_path, "LLM_PROVIDER=mistral\nMISTRAL_API_KEY=sk-from-file\n")

    settings = Settings(_env_file=env_file)

    assert settings.llm_provider == "mistral"
    assert settings.mistral_api_key is not None
    assert settings.mistral_api_key.get_secret_value() == "sk-from-file"


def test_environment_variable_overrides_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = write_env_file(tmp_path, "OLLAMA_MODEL=from-file\n")
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")

    assert Settings(_env_file=env_file).ollama_model == "from-env"


def test_empty_value_means_not_set(tmp_path: Path) -> None:
    env_file = write_env_file(tmp_path, "MISTRAL_API_KEY=\n")

    assert Settings(_env_file=env_file).mistral_api_key is None


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ValidationError, match="llm_provider"):
        Settings(_env_file=None)


def test_api_key_is_hidden_when_printed() -> None:
    settings = Settings(_env_file=None, mistral_api_key="sk-secret")

    assert "sk-secret" not in repr(settings)
    assert "sk-secret" not in str(settings)


def test_build_client_returns_ollama_by_default() -> None:
    client = build_client(Settings(_env_file=None, ollama_model="llama3.1:8b"))

    assert isinstance(client, OllamaClient)
    assert client.model == "llama3.1:8b"


def test_build_client_returns_mistral_when_selected() -> None:
    settings = Settings(_env_file=None, llm_provider="mistral", mistral_api_key="sk-test")

    client = build_client(settings)

    assert isinstance(client, MistralClient)
    assert client.model == "mistral-small-latest"


def test_build_client_requires_a_mistral_key() -> None:
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        build_client(Settings(_env_file=None, llm_provider="mistral"))
