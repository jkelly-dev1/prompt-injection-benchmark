"""The attack gate. Exits 1 when the BENCHMARK has stopped being able to measure.

WHAT THIS GATE PROTECTS, WHICH IS NOT WHAT MOST GATES PROTECT

It does not gate a product and it does not assert that injection is prevented.
It asserts that this repository can still tell defenses apart. A coverage matrix
degrades silently: payloads drift toward being blocked by everything or by
nothing, the table keeps rendering, every cell keeps showing a number, and the
whole thing stops ranking anything. That failure is invisible without a check
aimed directly at it.

IT ALWAYS RUNS AGAINST THE DETERMINISTIC MOCK PROVIDER, because a regression
gate has to be reproducible: the corpus pins which defense neutralizes which
payload, and a live model moves those around between runs. Real model behavior
belongs in SAMPLE_RUN.md, not in a pass or fail signal for CI.

THE CHECK THAT MATTERS MOST is gate_require_a_failing_defense. A benchmark in
which every measured defense reduces attack success is a benchmark that has
started flattering its subject. Prompt injection is not solved; if this suite
ever reports that all seven controls work, the corpus has been quietly tuned to
the defenses rather than the other way round, and that is the moment the project
stops being evidence. So "at least one defense must be measurably ineffective"
is a PASS condition, not a bug.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from bench.attacks.corpus import CORPUS, attack_classes, channels
from bench.audit import AuditLog
from bench.config import Settings, get_settings
from bench.harness import STANDARD_CONFIGS, run_matrix, total_trials
from bench.scoring import (
    effect_over_baseline,
    median_interval_width,
    noise_floor,
    per_config_noise,
    payloads_that_never_discriminate,
    summarize,
)


def _row(label: str, value: object) -> str:
    return f"  {label:<36} {value}"


def evaluate(settings: Settings | None = None) -> tuple[list[str], list[str]]:
    """(report lines, failure messages). Empty failures means the gate passes."""
    settings = settings or get_settings()
    # The gate never selects a real provider, whatever the environment says.
    settings = settings.model_copy(update={"agent_provider": "mock"})

    attempts = run_matrix(settings)
    reports = summarize(
        attempts,
        resamples=settings.bootstrap_resamples,
        confidence=settings.bootstrap_confidence,
    )
    floor = noise_floor(attempts)
    per_config = per_config_noise(attempts)
    dead, dead_rate = payloads_that_never_discriminate(attempts)
    effects = effect_over_baseline(reports, floor, per_config)
    baseline = reports.get(())

    lines: list[str] = []
    lines.append("Corpus")
    lines.append(_row("payloads", len(CORPUS)))
    lines.append(_row("attack_classes", len(attack_classes())))
    lines.append(_row("channels", len(channels())))
    lines.append(_row("configurations", len(STANDARD_CONFIGS)))
    lines.append(_row("repeats", settings.repeats))
    lines.append(_row("trials", total_trials(repeats=settings.repeats)))
    lines.append("")
    lines.append("Measurement quality")
    lines.append(_row("noise_floor", f"{floor:.3f}"))
    lines.append(_row("median_interval_width", f"{median_interval_width(reports):.3f}"))
    lines.append(
        _row("payloads_that_never_discriminate",
             f"{len(dead)}/{len(CORPUS)} ({dead_rate:.3f})")
    )
    if baseline is not None:
        lines.append("")
        lines.append("Undefended baseline (the number every effect is measured against)")
        lines.append(_row("attack_success", str(baseline.attack_success)))
        lines.append(_row("compliance", str(baseline.compliance)))
    lines.append("")
    lines.append("Effect of each configuration vs the undefended baseline")
    lines.append(_row("configuration", "reduction   flip   verdict"))
    for label, reduction, shown, applicable in effects:
        verdict = "shown" if shown else "NOT SHOWN (inside its own noise)"
        lines.append(
            _row(label[:34], f"{reduction:+.3f}      {applicable:.3f}  {verdict}")
        )

    failures: list[str] = []
    if len(CORPUS) < settings.gate_min_payloads:
        failures.append(
            f"corpus holds {len(CORPUS)} payloads, minimum is {settings.gate_min_payloads}"
        )
    if len(attack_classes()) < settings.gate_min_attack_classes:
        failures.append(
            f"corpus covers {len(attack_classes())} attack classes, minimum is "
            f"{settings.gate_min_attack_classes}"
        )
    if len(channels()) < settings.gate_min_channels:
        failures.append(
            f"corpus covers {len(channels())} channels, minimum is "
            f"{settings.gate_min_channels}"
        )
    if dead_rate > settings.gate_max_never_discriminate_rate:
        failures.append(
            f"payloads_that_never_discriminate {len(dead)}/{len(CORPUS)} = "
            f"{dead_rate:.3f} > {settings.gate_max_never_discriminate_rate:.3f}; the "
            f"corpus can no longer tell configurations apart. First few: "
            f"{', '.join(dead[:5])}"
        )
    if settings.gate_require_a_failing_defense:
        ineffective = [label for label, _reduction, shown, _floor in effects if not shown]
        if not ineffective:
            failures.append(
                "every measured configuration beat the baseline by more than the "
                "noise floor. Prompt injection is not solved, so this almost "
                "certainly means the corpus has been tuned to the defenses. See "
                "the gate module docstring."
            )
    return lines, failures


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    lines, failures = evaluate(settings)

    # The gate writes its audit trail to a temporary log so a CI run never
    # appends to the committed sample log.
    with tempfile.TemporaryDirectory() as directory:
        audit = AuditLog(Path(directory) / "gate.audit.jsonl")
        audit.append(
            "gate_decision",
            "gate",
            {"failures": failures, "passed": not failures},
        )
        chain_ok = audit.verify_chain()

    print("=" * 78)
    print("attack gate: deterministic mock agent, offline")
    print("=" * 78)
    print("\n".join(lines))
    print("")
    print(_row("audit_chain_intact", "yes" if chain_ok else "NO"))

    if failures or not chain_ok:
        print("\nATTACK GATE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        if not chain_ok:
            print("  - audit chain verification failed")
        return 1

    print(f"\nATTACK GATE PASSED ({len(CORPUS)} payloads)")
    print("  the corpus still discriminates between defense configurations, and at")
    print("  least one measured defense is still ineffective, which is the honest")
    print("  state of prompt-injection defense as of this writing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
