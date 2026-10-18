"""Unit tests for AnswerGenerator and LLM providers."""

import pytest
from rag_engine.answer_generator import (
    INSUFFICIENT_EVIDENCE_MARKER,
    AnswerGenerator,
    MockLLMProvider,
    OpenAILLMProvider,
)


def test_mock_llm_with_evidence():
    """Verify MockLLM returns answer with citation tag when evidence exists."""
    generator = AnswerGenerator(provider=MockLLMProvider())
    context = "--- BEGIN EVIDENCE [1] ---\nContent: Some academic finding\n--- END EVIDENCE [1] ---"

    answer = generator.generate_answer(query="What was found?", context_str=context)
    assert "[1]" in answer
    assert INSUFFICIENT_EVIDENCE_MARKER not in answer


def test_mock_llm_empty_context():
    """Verify empty context emits insufficient evidence response immediately."""
    generator = AnswerGenerator(provider=MockLLMProvider())
    answer = generator.generate_answer(query="What was found?", context_str="")
    assert INSUFFICIENT_EVIDENCE_MARKER in answer


def test_openai_provider_missing_key():
    """Verify OpenAI provider raises ValueError when api_key is empty."""
    provider = OpenAILLMProvider(api_key="")
    with pytest.raises(ValueError, match="API key is missing"):
        provider.generate(prompt="Hello")


def test_custom_fixed_answer():
    """Verify MockLLM supports overriding with custom fixed response for deterministic testing."""
    custom_mock = MockLLMProvider(fixed_answer="Custom verified answer [1].")
    generator = AnswerGenerator(provider=custom_mock)
    ans = generator.generate_answer("Query", "Context with [1]")
    assert ans == "Custom verified answer [1]."
