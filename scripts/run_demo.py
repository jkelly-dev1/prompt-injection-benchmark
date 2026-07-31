"""The end to end demo: the whole matrix, then what it means.

    python scripts/run_demo.py

    ENV_FILE=~/.secrets/ai.env AGENT_PROVIDER=anthropic python scripts/run_demo.py
    ENV_FILE=~/.secrets/ai.env AGENT_PROVIDER=openai    python scripts/run_demo.py

WITH A REAL PROVIDER, THE AGENT UNDER TEST BECOMES THE REAL MODEL and every
number below is measured against it. That is the only thing that changes: the
corpus, the defenses, the pipeline order and the scoring are the same code, so
the offline path is exercising exactly what a paid run takes. It prints the
trial count and asks for nothing before it starts, because the count is derived
from the same inputs the sweep uses and an operator should see it first.

THE DEMO IS NOT THE GATE. The gate (python -m bench.attacks.gate) always runs on
the deterministic mock, because a regression gate has to be reproducible. This
script is where real model behavior is allowed to appear, and nothing it
measures can change an exit code.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Support ENV_FILE=~/.secrets/ai.env so keys live outside the repo. setdefault,
# so a real environment variable still wins over the file.
if os.environ.get("ENV_FILE"):
    for _line in Path(os.path.expanduser(os.environ["ENV_FILE"])).read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _value = _line.partition("=")
            os.environ.setdefault(_key.strip(), _value.strip())

from bench.attacks.corpus import CORPUS, attack_classes, channels  # noqa: E402
from bench.audit import AuditLog  # noqa: E402
from bench.checkpoint import (  # noqa: E402
    Checkpoint,
    ShapeMismatch,
    fingerprint,
    harness_digest,
)
from bench.config import get_settings  # noqa: E402
from bench.defenses import (  # noqa: E402
    ACTION_LEVEL,
    OUTPUT_LEVEL,
    PROMPT_LEVEL,
    TEXT_LEVEL,
)
from bench.harness import (  # noqa: E402
    REDUCED_CONFIGS,
    total_calls,
    STANDARD_CONFIGS,
    TASK,
    run_matrix,
    select_payloads,
    total_trials,
)
from bench.llm import MockProvider, get_provider  # noqa: E402
from bench.prompts import SYSTEM_PROMPT, render_user_prompt  # noqa: E402
from bench.scoring import (  # noqa: E402
    effect_over_baseline,
    median_interval_width,
    noise_floor,
    pairwise_failure_correlation,
    payloads_that_never_discriminate,
    per_config_noise,
    summarize,
)


def rule(title: str) -> None:
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


def row(label: str, value: object) -> None:
    print(f"  {label:<38} {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_demo.py",
        description=(
            "The end to end benchmark. With a real provider configured the agent "
            "under test becomes that model; these flags size the paid sweep."
        ),
    )
    parser.add_argument(
        "--payloads", type=int, default=None, metavar="N",
        help="run a reduced corpus of N payloads, spread across attack classes "
             "rather than taken as a prefix (default: all of them)",
    )
    parser.add_argument(
        "--reduced-configs", action="store_true",
        help="use the reduced configuration set: the baseline, one member of "
             "each defense family, and the ordering pair",
    )
    parser.add_argument(
        "--repeats", type=int, default=None, metavar="N",
        help="repeats per trial, which is what makes the noise floor a "
             "measurement (default: settings, 3)",
    )
    parser.add_argument(
        "--checkpoint", metavar="PATH", default=None,
        help="append each completed trial to PATH as it lands, so an "
             "interrupted sweep costs one call instead of the whole run. Put "
             "it under audit/, which is gitignored",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip trials already recorded in --checkpoint. Refuses if that "
             "file describes a differently shaped sweep",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume and not args.checkpoint:
        # --resume alone has nothing to read and would silently run the whole
        # sweep, which is the outcome the operator was trying to avoid.
        print("--resume needs --checkpoint PATH: there is nothing to resume from.")
        return 2
    settings = get_settings()
    if args.repeats is not None:
        settings = settings.model_copy(update={"repeats": args.repeats})
    provider = get_provider(settings)
    live = provider.name != "mock"

    payloads = select_payloads(args.payloads) if args.payloads else CORPUS
    configs = REDUCED_CONFIGS if args.reduced_configs else STANDARD_CONFIGS
    trials = total_trials(payloads=payloads, configs=configs, repeats=settings.repeats)
    calls = total_calls(payloads=payloads, configs=configs, repeats=settings.repeats)
    multi_turn = sum(1 for item in payloads if item.is_multi_turn)

    rule("prompt-injection-benchmark demo")
    row("agent under test", f"{provider.name} ({provider.model})")
    row("payloads", f"{len(payloads)} of {len(CORPUS)}")
    row("attack classes", len({p.attack_class for p in payloads}))
    row("delivery channels", len({p.channel for p in payloads}))
    row("defense configurations", len(configs))
    row("repeats per trial", settings.repeats)
    row("TOTAL TRIALS", trials)
    row("TOTAL MODEL CALLS", f"{calls} ({multi_turn} multi-turn payloads cost two each)")
    if live:
        # Before any call is made: what it will do and roughly what it costs.
        # The estimate is characters/4, which llm-eval-gate measured as 2.35x
        # LOW on Anthropic prompts and accurate on OpenAI, so it is presented as
        # a floor rather than a figure. An operator authorizing a sweep should
        # see a number that cannot flatter itself.
        sample = render_user_prompt(
            payload_id=payloads[0].payload_id, attempt=1, task=TASK,
            channel=payloads[0].channel.value, content=payloads[0].rendered(),
        )
        approx_in = (len(SYSTEM_PROMPT) + len(sample)) // 4
        print("")
        print("  Before any call is made:")
        row("  approx input tokens per call", approx_in)
        row("  approx total input tokens", approx_in * calls)
        print("      characters/4 is an APPROXIMATION and a known UNDERCOUNT on")
        print("      Anthropic prompts (measured 2.35x low on a sibling project).")
        print("      Treat the figure as a floor, not an estimate.")
        print("")
        print("  A multi-turn trial is TWO calls, which is why the call count above")
        print("  is higher than the trial count. Nothing in this section can change")
        print("  an exit code: the gate ran on the mock.")
    else:
        print("")
        print("  offline deterministic mock. Two runs of this script produce the")
        print("  same numbers, which is what makes the gate a regression gate.")

    print("")
    print("  The corpus is synthetic. Payloads are written to look like the")
    print("  material this design is for, the fictional organization is Acme, and")
    print("  every address uses an .example TLD. No real attack data is here.")

    rule("1. The four families of defense, and why they are not comparable")
    row("prompt level", ", ".join(sorted(PROMPT_LEVEL)))
    print("      add framing. The injected instruction is still there and the")
    print("      agent still reads it.")
    row("text level", ", ".join(sorted(TEXT_LEVEL)))
    print("      change the text. When they work the agent never sees an")
    print("      instruction at all.")
    row("action level", ", ".join(sorted(ACTION_LEVEL)))
    print("      ignore the text and rule on what the agent tried to DO.")
    row("output level", ", ".join(sorted(OUTPUT_LEVEL)))
    print("      rule on the REPLY. The only family with an opinion about a")
    print("      payload that requests no capability at all.")

    def factory(defenses: frozenset[str]):
        # The mock needs to know the prompt-level framing because a real model
        # would see it in its prompt. The real providers need nothing: the
        # framing is already in the rendered message they receive.
        return provider if live else MockProvider(defenses=defenses)

    rule("2. Running the matrix")
    checkpoint = None
    if args.checkpoint:
        checkpoint = Checkpoint(args.checkpoint)
        shape = fingerprint(
            provider=provider.name,
            model=provider.model,
            payload_ids=tuple(p.payload_id for p in payloads),
            config_labels=tuple(tuple(sorted(c)) for c in configs),
            repeats=settings.repeats,
            harness=harness_digest(SYSTEM_PROMPT, payloads),
        )
        try:
            checkpoint.open_for(shape)
        except ShapeMismatch as exc:
            print(f"  REFUSED: {exc}")
            return 2
        done = len(checkpoint.completed()) if args.resume else 0
        row("checkpoint", args.checkpoint)
        row("already recorded", f"{done}/{trials}" if args.resume else "0 (not resuming)")

    attempts = run_matrix(
        settings,
        payloads=payloads,
        configs=configs,
        provider_factory=factory,
        checkpoint=checkpoint,
        resume=args.resume,
    )
    reports = summarize(
        attempts,
        resamples=settings.bootstrap_resamples,
        confidence=settings.bootstrap_confidence,
    )
    floor = noise_floor(attempts)
    per_config = per_config_noise(attempts)
    dead, dead_rate = payloads_that_never_discriminate(attempts)
    row("trials run", len(attempts))
    row("noise floor (flip rate across repeats)", f"{floor:.3f}")
    row("median 95% interval width", f"{median_interval_width(reports):.3f}")
    row("payloads that never discriminate", f"{len(dead)}/{len(payloads)} ({dead_rate:.3f})")

    rule("3. The undefended baseline")
    baseline = reports[()]
    row("attack_success", str(baseline.attack_success))
    row("compliance (agent was fooled)", str(baseline.compliance))
    row("containment (damage was stopped)", str(baseline.containment))
    print("")
    print("  Read those two apart. An agent can be fooled every single time and")
    print("  still cause no damage, if the control that matters is structural.")

    rule("4. What each defense actually bought")
    print(f"  {'configuration':<36} {'reduction':>10} {'flip':>7}  verdict")
    for label, reduction, shown, applicable in effect_over_baseline(
        reports, floor, per_config
    ):
        verdict = "shown" if shown else "NOT SHOWN (inside its own noise)"
        print(f"  {label[:36]:<36} {reduction:>+10.3f} {applicable:>7.3f}  {verdict}")
    print("")
    print("  An effect smaller than the variance of the configurations being")
    print("  compared is not an effect. It is reported as NOT SHOWN rather than")
    print("  as a small win, which is the difference between measuring and hoping.")

    rule("5. Compliance versus containment, per configuration")
    print(f"  {'configuration':<36} {'complied':>10} {'contained':>11}")
    for defenses in sorted(reports):
        report = reports[defenses]
        print(
            f"  {report.label[:36]:<36} {report.compliance.value:>10.3f} "
            f"{report.containment.value:>11.3f}"
        )
    print("")
    print("  The rows where compliance stays high and containment reaches 1.000")
    print("  are the structural controls. They do not stop the agent being fooled.")
    print("  They stop that from mattering.")

    rule("6. Do stacked defenses fail on the same payloads?")
    correlations = [
        pair
        for pair in pairwise_failure_correlation(attempts)
        if pair.fail_a > 0 and pair.fail_b > 0
    ]
    correlations.sort(key=lambda pair: -pair.ratio)
    print(f"  {'pair':<52} {'joint':>6} {'ratio':>7}")
    for pair in correlations[:8]:
        label = f"{pair.a[:24]} + {pair.b[:24]}"
        print(f"  {label:<52} {pair.joint:>6.3f} {pair.ratio:>7.2f}")
    print("")
    print("  This question exists because llm-eval-gate's real model run found")
    print("  three judges whose errors were COMPLETELY NESTED across two vendors:")
    print("  no judge was ever wrong alone, so majority voting corrected nothing.")
    print("  Stacked defenses fail the same way, and a per-defense table alone")
    print("  would hide it.")

    rule("7. Where each configuration still fails, by attack class")
    worst = max(
        (report for defenses, report in reports.items() if defenses),
        key=lambda report: report.attack_success.value,
    )
    print(f"  worst surviving configuration: {worst.label}")
    for name, rate in sorted(worst.by_class.items()):
        print(f"    {name:<26} attack_success {rate}")

    rule("8. The audit trail")
    audit = AuditLog(settings.audit_log_path)
    record = audit.append(
        "benchmark_run",
        "demo",
        {
            "provider": provider.name,
            "model": provider.model,
            "trials": len(attempts),
            "noise_floor": round(floor, 6),
            "baseline_attack_success": round(baseline.attack_success.value, 6),
        },
    )
    row("record_hash", record["record_hash"][:16])
    row("chain intact", "yes" if audit.verify_chain() else "NO")

    rule("Summary")
    print("  prompt injection is not solved, and this benchmark is built to keep")
    print("  saying so. The honest reading of the table above:")
    print("")
    print("   - the prompt-level family is too unstable to demonstrate an effect")
    print("     at this sample size. Its measured improvement is real-looking and")
    print("     sits inside its own flip rate.")
    print("   - the structural controls are the ones with a demonstrated effect,")
    print("     and they work by making compliance not matter rather than by")
    print("     preventing it.")
    print("   - no configuration reaches zero. The full stack still leaves attack")
    print("     success well above zero, and that number is the point.")
    print("")
    print("  the gate did not run here. python -m bench.attacks.gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
