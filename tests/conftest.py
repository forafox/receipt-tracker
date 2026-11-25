import pytest

from extract import agent


@pytest.fixture(autouse=True)
def _offline_language_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests away from Ollama and OpenKilo, whatever runs on the developer's machine.

    Tests of the agent itself turn these back on with their own ``monkeypatch`` calls.
    """
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    monkeypatch.delenv("KILO_TOKEN", raising=False)
    monkeypatch.setattr(agent, "_local_model_available", False)
