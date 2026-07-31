"""Shared fixtures, and the one autouse guard the whole suite depends on.

The guard matters more than the fixtures. Every test in this repository is
offline, and the way that stops being true is not a deliberate change: it is a
developer with ANTHROPIC_API_KEY exported in their shell running pytest and not
noticing that a sweep of 1,296 trials just went to a paid API. So the provider
environment is cleared for every test regardless of what the ambient shell says,
and a test that wants a credentialed provider constructs Settings explicitly.

There is no recorded-fixture or replay layer here on purpose. The offline agent
is a synthesizing mock that is deterministic by construction (see bench/llm.py),
so there is nothing to record and nothing to go stale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every environment variable that could route a test at a real provider.
_PROVIDER_ENV = (
    "AGENT_PROVIDER",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_MODEL",
    "OPENAI_MODEL",
    "AGENT_MODEL",
    "ENV_FILE",
)


@pytest.fixture(autouse=True)
def force_offline(monkeypatch):
    """Clear provider credentials for every test, without exception.

    autouse because opting in per test is the same as not having it: the test
    that forgets is exactly the test that would spend money.
    """
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def offline_settings() -> Settings:
    """Settings pinned to the deterministic mock, with both keys absent."""
    return Settings(
        agent_provider="mock",
        anthropic_api_key=None,
        openai_api_key=None,
    )
