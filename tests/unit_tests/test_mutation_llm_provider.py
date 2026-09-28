"""The mutation engine's LLM provider is selectable (no network: models are only constructed)."""
import pytest
from langchain_google_genai import ChatGoogleGenerativeAI

from src.behavioral_engine.moe_score_subgraph.llm_judges.meta_judge import _AutoToolChoiceChatOpenAI
from src.mutation_engine.nodes.invoke_mutation_llm import _build_mutation_model


def test_default_provider_is_gemini(monkeypatch):
    monkeypatch.delenv("MUTATION_ENGINE_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    assert isinstance(_build_mutation_model(), ChatGoogleGenerativeAI)


def test_meta_provider_uses_muse_spark(monkeypatch):
    monkeypatch.setenv("MUTATION_ENGINE_PROVIDER", "meta")
    monkeypatch.setenv("MODEL_API_KEY", "test-meta-key")
    monkeypatch.setenv("MUTATION_ENGINE_MODEL", "gemini-2.5-flash")  # must not leak into the Meta model
    monkeypatch.delenv("MUTATION_ENGINE_META_MODEL", raising=False)
    model = _build_mutation_model()
    assert isinstance(model, _AutoToolChoiceChatOpenAI)  # handles Meta's tool_choice="auto" restriction
    assert model.model_name == "muse-spark-1.2"


def test_meta_model_override(monkeypatch):
    monkeypatch.setenv("MUTATION_ENGINE_PROVIDER", "meta")
    monkeypatch.setenv("MODEL_API_KEY", "test-meta-key")
    monkeypatch.setenv("MUTATION_ENGINE_META_MODEL", "muse-spark-1.3")
    assert _build_mutation_model().model_name == "muse-spark-1.3"


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("MUTATION_ENGINE_PROVIDER", "other")
    with pytest.raises(ValueError):
        _build_mutation_model()
