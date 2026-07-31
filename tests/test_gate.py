"""The attack gate, which protects the BENCHMARK rather than a product.

The gate never asserts that prompt injection is prevented. It asserts that this
repository can still tell defense configurations apart, and a coverage matrix
loses that ability silently: payloads drift toward being blocked by everything
or by nothing, every cell keeps printing a number, and the table stops ranking
anything. So the properties asserted here are the ones that make a red build
possible at all.

Two of them are worth naming. The gate ALWAYS runs against the deterministic
mock, whatever the environment says and whatever credentials are lying around,
because a regression signal that changes with a live model's mood is not a
signal and because a sweep of over a thousand trials pointed at a paid API is a
real bill. That is asserted with the network taken away entirely, not by reading
the code.

And `gate_require_a_failing_defense` is a PASS condition, not a bug: a benchmark
in which every measured control works has started flattering its subject. The
test for it deliberately simulates the flattering run and asserts that the build
goes red, because the day that check stops firing is the day the project stops
being evidence.
"""

from __future__ import annotations

import socket

from bench.attacks import gate
from bench.attacks.corpus import CORPUS
from bench.attacks.gate import evaluate, main
from bench.config import Settings


def _settings(**overrides) -> Settings:
    """Settings built explicitly so an .env file cannot move a gate outcome.

    `bootstrap_resamples` is lowered because nothing the gate FAILS on is
    computed from an interval: the failure conditions read corpus sizes, the
    never-discriminate rate and point estimates of attack success. Keeping the
    committed default here would cost a couple of seconds per call to compute
    bounds no assertion in this module looks at.
    """
    base = {
        "agent_provider": "mock",
        "anthropic_api_key": None,
        "openai_api_key": None,
        "bootstrap_resamples": 200,
    }
    base.update(overrides)
    return Settings(**base)


def test_the_committed_corpus_passes_the_gate():
    """Green on the defaults, or every failure asserted below proves nothing.

    Run with no argument, so this is the committed configuration and the
    committed corpus exactly as CI sees them. The report body is checked as well
    as the failure list, because a gate that returns no failures while printing
    nothing measurable would satisfy the first assertion alone.
    """
    lines, failures = evaluate()

    assert failures == []
    text = "\n".join(lines)
    assert "Corpus" in text
    assert "Measurement quality" in text
    assert "payloads_that_never_discriminate" in text
    assert "Undefended baseline" in text
    assert "Effect of each configuration vs the undefended baseline" in text
    assert f"payloads                             {len(CORPUS)}" in text
    # The honest state of the art, printed rather than merely computed: at least
    # one measured configuration is still inside its own noise.
    assert "NOT SHOWN (inside its own noise)" in text


def test_the_gate_runs_on_the_mock_even_when_a_real_provider_is_fully_configured(
    monkeypatch,
):
    """A safety property: a stray key must not be able to bill a sweep.

    The settings here name Anthropic AND carry a key, which is the exact state
    in which `llm.get_provider` would hand back a live client. The gate copies
    its settings to the mock before running anything, so the sweep stays
    offline.

    That is asserted by removing the network rather than by inspecting a field:
    `socket.socket` is replaced with a tripwire, and the real provider classes
    and the provider selector are replaced with tripwires of their own. If any
    part of the gate reached for a live model, one of the four would fire.

    The two runs are then compared line for line. Identical output is the strong
    form of the claim, since a gate that fell back to the mock but reported
    something different would still be unreproducible.
    """

    def no_network(*args, **kwargs):
        raise AssertionError("the gate opened a socket")

    def no_real_provider(*args, **kwargs):
        raise AssertionError("the gate selected a real provider")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr("bench.llm.get_provider", no_real_provider)
    monkeypatch.setattr("bench.llm.AnthropicProvider", no_real_provider)
    monkeypatch.setattr("bench.llm.OpenAIProvider", no_real_provider)

    configured = _settings(agent_provider="anthropic", anthropic_api_key="sk-test")
    assert configured.agent_provider == "anthropic"
    assert configured.anthropic_api_key == "sk-test"

    lines, failures = evaluate(configured)
    assert failures == []

    mock_lines, mock_failures = evaluate(_settings())
    assert failures == mock_failures
    assert lines == mock_lines, (
        "naming a provider must not change a single line of the gate's report"
    )


def test_every_defense_beating_its_floor_fails_the_gate(monkeypatch):
    """The honesty control, simulated on the run it is meant to catch.

    Prompt injection is not solved. If this suite ever reports that all seven
    controls measurably work, the likeliest explanation by far is that the
    corpus has been tuned to the defenses, so "at least one measured defense is
    ineffective" is a pass condition. The flattering run is simulated by
    replacing `effect_over_baseline` with one that returns nothing but shown
    rows, and the gate has to go red on it.

    Mutation check, executed in-test: with `gate_require_a_failing_defense` set
    to False the same all-shown run produces no failure at all. That is what an
    unguarded benchmark looks like from the outside, a clean green build on a
    matrix that has quietly stopped disagreeing with its author.
    """

    def everything_works(reports, floor, per_config=None):
        return [
            ("delimiter_fencing", 0.62, True, 0.01),
            ("egress_filter", 0.55, True, 0.00),
            ("input_pattern_filter", 0.41, True, 0.00),
            ("tool_allowlist", 0.33, True, 0.00),
        ]

    monkeypatch.setattr(gate, "effect_over_baseline", everything_works)

    lines, failures = evaluate(_settings())
    assert any(
        "every measured configuration beat the baseline" in failure
        for failure in failures
    ), failures
    assert "NOT SHOWN (inside its own noise)" not in "\n".join(lines), (
        "the simulated run really does report every defense as working"
    )

    # The mutation: turn the honesty control off and the flattering run passes.
    _lines, unguarded = evaluate(
        _settings(gate_require_a_failing_defense=False)
    )
    assert unguarded == [], (
        "without the check, a benchmark where everything works is a green build"
    )


def test_a_corpus_below_the_minimum_payload_count_fails_the_gate(monkeypatch):
    """A matrix over a handful of payloads has intervals as wide as the table.

    The minimum is a floor on how thin the corpus may be spread before the
    numbers stop resolving anything, and it fails loudly rather than printing a
    matrix of cells holding two observations each.
    """
    monkeypatch.setattr(gate, "CORPUS", CORPUS[:3])

    settings = _settings()
    _lines, failures = evaluate(settings)

    assert any("corpus holds 3 payloads" in failure for failure in failures)
    assert any(
        f"minimum is {settings.gate_min_payloads}" in failure
        for failure in failures
    )


def test_a_corpus_that_stopped_discriminating_fails_the_gate(monkeypatch):
    """The vacuous benchmark detector, which is the failure a matrix hides best.

    A payload with the same outcome under every configuration ranks nothing, and
    a corpus made of those renders a beautifully formatted table that cannot
    tell one defense from another. The real corpus reports 0.000 here, so the
    saturated case is simulated: every payload dead, a rate of 1.000, well over
    the configured ceiling.
    """
    dead_ids = sorted(payload.payload_id for payload in CORPUS)

    def everything_is_dead(attempts):
        return dead_ids, 1.0

    monkeypatch.setattr(
        gate, "payloads_that_never_discriminate", everything_is_dead
    )

    settings = _settings()
    _lines, failures = evaluate(settings)

    assert any(
        "payloads_that_never_discriminate" in failure
        and "can no longer tell configurations apart" in failure
        for failure in failures
    ), failures
    assert settings.gate_max_never_discriminate_rate < 1.0

    # The unsimulated corpus is on the right side of the same threshold, so the
    # check is a live one rather than a permanently satisfied constant.
    monkeypatch.undo()
    _lines, real_failures = evaluate(settings)
    assert real_failures == []


def test_main_returns_zero_and_prints_a_readable_verdict(capsys):
    """CI reads an integer, so the exit code is asserted before the prose.

    A gate that prints PASSED and returns non-zero, or the reverse, is not a
    gate. The audit line is checked too, because `main` writes its trail to a
    temporary log and a broken chain is one of the two things that can turn this
    run red without any failure message being produced.
    """
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "attack gate: deterministic mock agent, offline" in output
    assert f"ATTACK GATE PASSED ({len(CORPUS)} payloads)" in output
    assert "audit_chain_intact                   yes" in output
    assert "ATTACK GATE FAILED" not in output
