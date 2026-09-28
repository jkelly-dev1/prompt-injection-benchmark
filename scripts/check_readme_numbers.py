"""Re-derive the published figures and diff them against README.md and SAMPLE_RUN.md.

A README is prose and drifts; the per-trial rows under `results/` are evidence
and do not. This script rebuilds each real-model figure from those rows, and
each offline figure from a fresh run of the mock matrix (the code the gate
runs), and asserts the exact string appears in the document that quotes it,
so a change that shifts a figure fails loudly instead of leaving the
document silently wrong.

    python3 scripts/check_readme_numbers.py                  check both documents
    python3 scripts/check_readme_numbers.py --emit           print the README.md rows
    python3 scripts/check_readme_numbers.py --emit SAMPLE_RUN.md
                                                             print that document's rows

Each row is anchored to its label or its sentence and must match exactly
once: a bare number is also some other figure, and a row that matches two
places leaves one of them free to change.

Figures on README.md this script does not derive, named so the omission is
visible: the test counts in the Quickstart (164 and 162), which pytest
states; the "blocked 100%" a block-rate report prints; the +0.194 an earlier
README printed for the pattern filter; the 0.250 undefended baseline from a
reduced sweep whose rows are not published; and the sibling projects'
figures (29 of 40, 24 of those 29, 30 of 30), which belong to those
repositories. On SAMPLE_RUN.md the reduced-sweep capture and the list of
reduced-sweep baselines are not derived, for the same reason.

It reads only what a reader gets. The raw checkpoints under `audit/` are not
shipped, so deriving from them would prove nothing about the published tree.
Every real-model figure comes from `results/`, which is tracked.

The count is printed whether or not anything is missing, so a version of this
script that silently stopped deriving half of them is visible in its own output
rather than reported as clean.
"""

import functools
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import rescore  # noqa: E402
from bench import harness  # noqa: E402
from bench.attacks.corpus import CORPUS, attack_classes, channels  # noqa: E402
from bench.config import Settings  # noqa: E402
from bench.defenses import (  # noqa: E402
    ALLOWED_HOSTS,
    ALLOWED_TOOLS,
    PROMPT_DEFENSE_CEILING,
    PROMPT_LEVEL,
    Request,
    apply_transforms,
    target_host,
    tool_name,
)
from bench.scoring import (  # noqa: E402
    effect_over_baseline,
    median_interval_width,
    noise_floor,
    payloads_that_never_discriminate,
    per_config_noise,
    summarize,
)

OPUS, TERRA = "claude-opus-5", "gpt-5.6-terra"
README, SAMPLE_RUN = "README.md", "SAMPLE_RUN.md"
GUARD = ("output_provenance_guard",)

#: The rows of the README's three-column defense table.
EFFECT_ROWS = ("delimiter_fencing", "provenance_tagging",
               "instruction_hierarchy", "egress_filter", "tool_allowlist",
               "egress_filter+tool_allowlist")

#: Where each SAMPLE_RUN.md row is looked for: the text between two headings.
REGIONS = {
    "opus": ("## Real model run (Anthropic, claude-opus-5)",
             "The superseded 288-trial reduced sweep used:"),
    "terra": ("## Real model run (OpenAI, gpt-5.6-terra)",
              "## What the real model runs found"),
    "found": ("## What the real model runs found", None),
}

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n):
    """A count under one hundred spelled the way the documents spell it."""
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + ("" if not ones else " " + _ONES[ones])


def normalize(text):
    """Collapse whitespace and emphasis so a reflow is not a false alarm."""
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", "")).strip()


def _stats(trials, resamples):
    reports = summarize(trials, resamples=resamples)
    floor = noise_floor(trials)
    per_config = per_config_noise(trials)
    effect_rows = effect_over_baseline(reports, floor, per_config)
    dead, dead_rate = payloads_that_never_discriminate(trials)
    return {
        "attempts": trials,
        "trials": len(trials),
        "reports": reports,
        "floor": floor,
        "width": median_interval_width(reports),
        "effect_rows": effect_rows,
        "effects": {row[0]: row for row in effect_rows},
        "dead": dead,
        "dead_rate": dead_rate,
        "payloads": len({t.payload_id for t in trials}),
    }


def measured(model, rescored=True):
    """Every figure the documents quote for `model`: the published rows, with
    results/rescored.<model>.json applied when `rescored` is true."""
    trials = rescore.attempts(model, rescored=rescored)
    # The printed captures quote intervals, so the printed rows are summarized
    # at the committed resample count; the RE-SCORED tables quote only point
    # values, which do not depend on it.
    out = _stats(trials, _mock_settings().bootstrap_resamples if not rescored
                 else 2)
    reports = out["reports"]
    complied = [t for t in trials if t.complied]
    kinds = {}
    for t in complied:
        kinds[t.action.kind.value] = kinds.get(t.action.kind.value, 0) + 1
    by_payload = {}
    for t in trials:
        by_payload.setdefault(t.payload_id, []).append(t)
    gets = [t.action.target for t in complied if t.action.kind.value == "http_get"]
    tools = [t for t in complied if t.action.kind.value == "tool_call"]
    all_tool_calls = [t for t in trials if t.action.kind.value == "tool_call"]
    out.update({
        "baseline": reports[()].compliance.value,
        "baseline_complied": reports[()].compliance.successes,
        "baseline_n": reports[()].compliance.n,
        "all": len(complied) / len(trials),
        "complied": len(complied),
        "complied_payloads": sorted({t.payload_id for t in complied}),
        "answer_shaped": kinds.get("answer", 0),
        "action_taking": len(complied) - kinds.get("answer", 0),
        "kinds": kinds,
        "action_classes": {t.attack_class.value for t in complied
                           if t.action.kind.value != "answer"},
        "gets_allowed": sum(target_host(g) in ALLOWED_HOSTS for g in gets),
        "gets_allowed_payloads": sorted({
            t.payload_id for t in complied if t.action.kind.value == "http_get"
            and target_host(t.action.target) in ALLOWED_HOSTS}),
        "tools_granted": sum(tool_name(t.action.target) in ALLOWED_TOOLS
                             for t in tools),
        "tool_calls": all_tool_calls,
        "textless": sum(t.action.refused_without_text for t in trials),
        "with_text": sum(1 for t in trials if t.action.answer),
        "complied_with_text": sum(1 for t in complied if t.action.answer),
        "never_a_word": sum(all(t.action.refused_without_text for t in group)
                            for group in by_payload.values()),
        "calls": sum(2 if rescore.PAYLOADS[t.payload_id].is_multi_turn else 1
                     for t in trials),
        "zero_compliance": sorted(r.label for d, r in reports.items()
                                  if d and r.compliance.value == 0),
        "zero_success": sorted(r.label for d, r in reports.items()
                               if d and r.attack_success.value == 0),
    })
    return out


def _mock_settings():
    """The committed defaults, set explicitly so an .env cannot move a figure."""
    defaults = {name: Settings.model_fields[name].default
                for name in ("repeats", "bootstrap_resamples",
                             "bootstrap_confidence")}
    return Settings(agent_provider="mock", anthropic_api_key=None,
                    openai_api_key=None, **defaults)


def offline():
    """The offline figures: the mock matrix, run the way the gate runs it."""
    settings = _mock_settings()
    out = _stats(harness.run_matrix(settings), settings.bootstrap_resamples)
    out["confidence"] = settings.bootstrap_confidence
    out["repeats"] = settings.repeats
    out["zero_compliance"] = [r.label for d, r in out["reports"].items()
                              if d and r.compliance.value == 0]
    out["class_n"] = sorted({rate.n for rate in out["reports"][()]
                             .by_class.values()})
    out["plant_turn_guard"] = _guard_on_the_plant_turn(settings)
    return out


def _guard_on_the_plant_turn(settings):
    """The output guard's offline reduction if it were handed the PLANT turn's
    material on every payload: the counterfactual the README prices."""
    def plant_turn(payload, defenses):
        request = Request(task=harness.TASK, untrusted=payload.rendered())
        return apply_transforms(request, defenses).untrusted

    shipped = harness._screened_material
    harness._screened_material = plant_turn
    try:
        trials = harness.run_matrix(
            settings, configs=(frozenset(), frozenset(GUARD)))
    finally:
        harness._screened_material = shipped
    reports = summarize(trials, resamples=2)
    return (reports[()].attack_success.value
            - reports[GUARD].attack_success.value)


def _guard_cross_turn_share():
    """How much of the output guard's gpt-5.6-terra containment the cross-turn
    fix removed: the re-scored containment against the same trials with only
    the contained-only moves put back."""
    moved = {rescore.key(m): m for m in rescore.overlay(TERRA)}
    held = total = 0
    for row in rescore.rows(TERRA):
        if tuple(row["defenses"]) != GUARD:
            continue
        m = moved.get(rescore.key(row))
        verdict = m["rescored"] if m else row
        if not verdict["complied"]:
            continue
        total += 1
        contained_only = m and m["printed"]["complied"] == verdict["complied"]
        held += (m["printed"] if contained_only else verdict)["contained"]
    return held / total


def _cell(effect, with_verdict):
    """One cell of the three-column table. The offline and claude-opus-5
    columns print a value only when it is shown and then omit the verdict;
    the gpt-5.6-terra column prints both, always."""
    _, reduction, shown, _ = effect
    if with_verdict:
        return "%+.3f %s" % (reduction, "shown" if shown else "NOT SHOWN")
    return "%+.3f" % reduction if shown else "NOT SHOWN"


def _row(tag, text, region=None):
    """A row whose text is literal: the pattern is the normalized text."""
    text = normalize(text)
    return (tag, re.escape(text), text, region)


def _block(tag, lines, region):
    """A row that is a run of printed lines, matched as one string."""
    return _row(tag, " ".join(lines), region)


def _same_size(o, t):
    """Whether both sweeps ran the same trials and calls, which is what the
    sentences stating one size for both assume."""
    return (o["trials"], o["calls"]) == (t["trials"], t["calls"])


@functools.lru_cache(maxsize=None)
def derive():
    """{document: [(tag, pattern, text, region)]}. Each pattern is a regular
    expression over the normalized document, or over the normalized region
    of it named by `region`; `text` is what --emit prints."""
    o, t = measured(OPUS), measured(TERRA)
    op, tp = measured(OPUS, rescored=False), measured(TERRA, rescored=False)
    m = offline()
    return {README: _readme_rows(o, t, op, tp, m),
            SAMPLE_RUN: _sample_run_rows(o, t, op, tp)}


def _readme_rows(o, t, op, tp, m):
    lit = re.escape
    moved_t, moved_o = rescore.overlay(TERRA), rescore.overlay(OPUS)
    rejudged = sum(not x["needs_answer_text"] for x in moved_t)
    n = f"{o['trials']:,}"
    sweep = n if _same_size(o, t) else "(the two sweeps differ)"
    calls = f"{o['calls']:,}" if _same_size(o, t) else "(differ)"
    zero_everywhere = not (m["zero_compliance"] or t["zero_compliance"])
    ceiling_series = sum(PROMPT_DEFENSE_CEILING * (0.5 ** i) / 2
                         for i in range(len(PROMPT_LEVEL)))
    eff = m["effects"]
    out = [
        # The opening paragraphs.
        _row("corpus", "%s synthetic payloads across %s delivery channels and "
             "%s attack classes are run against %s independently toggleable "
             "defense configurations, %s times each"
             % (words(len(CORPUS)).capitalize(), words(len(channels())),
                words(len(attack_classes())), words(len(m["reports"])),
                words(m["repeats"]))),
        _row("opus:trials", "%s nothing cleared the floor at all, because "
             "that model complied %d times in %s trials"
             % (OPUS, o["complied"], n)),
        _row("terra:complied", "%s complied %d times in the same %s"
             % (TERRA, t["complied"], n)),
        _row("terra:printed", "the terra run printed %d, and the section below "
             "explains the %s" % (tp["complied"],
                                  words(t["complied"] - tp["complied"]))),
        _row("zero-compliance", "%s drives it to zero offline or on %s; %s do "
             "on %s, which complied on %.3f of undefended trials"
             % ("No configuration" if zero_everywhere else "Some configuration",
                TERRA, words(len(o["zero_compliance"])), OPUS, o["baseline"])),
    ]
    # The offline capture: the header block and the twelve-row table.
    b = m["reports"][()]
    out += [
        _row("offline:header", "%d payloads %d delivery channels %d attack "
             "classes %d configurations %d repeats %s trials"
             % (len(CORPUS), len(channels()), len(attack_classes()),
                len(m["reports"]), m["repeats"], f"{m['trials']:,}")),
        _row("offline:baseline", "undefended baseline attack_success %s "
             "compliance %.3f containment %.3f noise floor %.3f (flip rate "
             "across repeats) median %d%% interval %.3f payloads that never "
             "discriminate %d/%d"
             % (b.attack_success, b.compliance.value, b.containment.value,
                m["floor"], round(m["confidence"] * 100), m["width"],
                len(m["dead"]), m["payloads"])),
    ]
    for label, reduction, shown, flip in m["effect_rows"]:
        name = (lit(label) if len(label) <= 34
                else lit(label[:20]) + r"[^ ]*\.\.\.")
        verdict = "shown" if shown else "NOT SHOWN"
        out.append(("offline:%s" % label,
                    r"(?:^| )%s %s %s %s(?: |$)"
                    % (name, lit("%+.3f" % reduction), lit("%.3f" % flip),
                       verdict),
                    "%s %+.3f %.3f %s" % (label, reduction, flip, verdict),
                    None))
    fencing = eff["delimiter_fencing"]
    tools, egress = eff["tool_allowlist"], eff["egress_filter"]
    pair = eff["egress_filter+tool_allowlist"]
    summed = "%.3f" % (tools[1] + egress[1]) == "%.3f" % pair[1]
    out += [
        _row("offline:fencing", "Fencing measures %+.3f, which looks like a "
             "real improvement until you notice its flip rate is %.3f"
             % (fencing[1], fencing[3])),
        _row("offline:additive", "tool_allowlist buys %+.3f and egress_filter "
             "buys %+.3f, and together they buy %+.3f, %s"
             % (tools[1], egress[1], pair[1],
                "exactly the sum" if summed else "not the sum")),
        _row("offline:unicode", "unicode_normalization measures %+.3f on its "
             "own" % eff["unicode_normalization"][1]),
        _row("offline:pair", "the pair measures %+.3f against the filter's "
             "own %+.3f" % (eff["input_pattern_filter+unicode_normalization"][1],
                            eff["input_pattern_filter"][1])),
        _row("offline:guard", "It measures %+.3f." % eff[GUARD[0]][1]),
    ]
    # Real model results.
    out += [
        _row("sweeps", "Two full sweeps, %s trials and %s model calls each"
             % (sweep, calls)),
        _row("answer-shaped", "for the %d answer-shaped compliances, %d from "
             "%s and %d from %s" % (o["answer_shaped"] + t["answer_shaped"],
                                    t["answer_shaped"], TERRA,
                                    o["answer_shaped"], OPUS)),
        _row("action-taking", "for the other %d (RE-SCORED; the run printed "
             "%d)" % (t["action_taking"], tp["action_taking"])),
        _row("moved", "which moved %s %s trials and %s %s trial"
             % (words(len(moved_t)), TERRA,
                "no" if not moved_o else words(len(moved_o)), OPUS)),
        _row("rows:opus", "results/trials.claude-opus-5.jsonl %s trials"
             % f"{len(rescore.rows(OPUS)):,}"),
        _row("rows:terra", "results/trials.gpt-5_6-terra.jsonl %s trials"
             % f"{len(rescore.rows(TERRA)):,}"),
        _row("overlay-files", "results/rescored.gpt-5_6-terra.json (%s "
             "trials) and results/rescored.claude-opus-5.json (%s)"
             % (words(len(moved_t)), "none" if not moved_o
                else words(len(moved_o)))),
        _row("overlay-split", "%s of the %s moved verdicts read only the "
             "recorded action, and a test re-judges them, and every other "
             "such trial, from the rows. The other %s read answer text"
             % (words(rejudged).capitalize(), words(len(moved_t)),
                words(len(moved_t) - rejudged))),
        _row("overlay-unchecked", "so %s of the %s re-scored verdicts cannot "
             "be checked from the rows" % (words(len(moved_t) - rejudged),
                                           words(len(moved_t)))),
        # The RE-SCORED summary table, one row at a time, both columns.
        _row("table:baseline", "undefended baseline compliance %.3f %.3f"
             % (o["baseline"], t["baseline"])),
        _row("table:all", "compliance, all configurations %.3f %.3f"
             % (o["all"], t["all"])),
        _row("table:action", "of which action-taking %d %d"
             % (o["action_taking"], t["action_taking"])),
        _row("table:textless", "refused, no text %.3f %.3f"
             % (o["textless"] / o["trials"], t["textless"] / t["trials"])),
        _row("table:floor", "noise floor %.3f %.3f" % (o["floor"], t["floor"])),
        _row("table:dead", "payloads that never discriminate %d/%d %d/%d"
             % (len(o["dead"]), o["payloads"], len(t["dead"]), t["payloads"])),
    ]
    out += _opus_prose(o, README) + _terra_prose(t, README)
    # The three-column defense table, every column exact.
    for name in EFFECT_ROWS:
        text = "%s %s %s %s" % (name, _cell(eff[name], False),
                                _cell(o["effects"][name], False),
                                _cell(t["effects"][name], True))
        out.append(("effect:%s" % name, r"(?:^| )%s(?: |$)" % lit(text),
                    text, None))
    hier, hier_p = t["effects"]["instruction_hierarchy"], \
        tp["effects"]["instruction_hierarchy"]
    uni, uni_p = t["effects"]["unicode_normalization"], \
        tp["effects"]["unicode_normalization"]
    ouni, ouni_p = o["effects"]["unicode_normalization"], \
        op["effects"]["unicode_normalization"]
    out += [
        _row("opus:floor", "with %d compliances in %s trials there is almost "
             "no attack success to reduce, so no configuration can clear even "
             "a %.3f floor" % (o["complied"], n,
                               min(e[3] for e in o["effect_rows"]))),
        _row("terra:hierarchy", "%+.3f RE-SCORED, %+.3f PRINTED, against a "
             "per-configuration floor of %.3f under %s"
             % (hier[1], hier_p[1], hier[3],
                "both scorings" if hier[3] == hier_p[3] else "the re-scoring")),
        _row("terra:fencing", "Fencing the untrusted material cut compliance "
             "from %.3f to %.3f" % (
                 t["baseline"],
                 t["reports"][("delimiter_fencing",)].compliance.value)),
        _row("terra:containment", "egress_filter contained %.3f of the "
             "compliances it saw while tool_allowlist contained %.3f"
             % (t["reports"][("egress_filter",)].containment.value,
                t["reports"][("tool_allowlist",)].containment.value)),
        _row("unicode", "unicode_normalization alone buys nothing anywhere: "
             "%+.3f RE-SCORED on %s (%+.3f PRINTED), %+.3f on %s under %s, "
             "and %+.3f offline"
             % (uni[1], TERRA, uni_p[1], ouni[1], OPUS,
                "both scorings" if "%.3f" % ouni[1] == "%.3f" % ouni_p[1]
                else "the re-scoring", eff["unicode_normalization"][1])),
        _row("zeros", "on %s %s configurations measure 0.000 attack success "
             "and 0.000 compliance" % (
                 OPUS, words(len(o["zero_compliance"]))
                 if o["zero_compliance"] == o["zero_success"] else "(differ)")),
        _row("terra:zero", "on %s the %s reaches 0.000 attack success"
             % (TERRA, "full stack" if t["zero_success"] == [max(
                 (r.label for r in t["reports"].values()),
                 key=lambda label: label.count("+"))]
                else "(another set)")),
        _row("opus:four-fewer", "opus complied on %.3f of undefended trials, "
             "so a zero there is %s fewer compliances out of %d"
             % (o["baseline"], words(o["baseline_complied"]),
                o["baseline_n"])),
        _row("ceiling", "at %s defenses the series totals %s against a %s "
             "ceiling" % (words(len(PROMPT_LEVEL)), repr(round(ceiling_series, 10)),
                          repr(PROMPT_DEFENSE_CEILING))),
        _row("offline:sizes", "At the default sizes that is %s trials. The "
             "multi-turn payloads cost two calls each, so a full sweep is %s "
             "model calls" % (f"{harness.total_trials():,}",
                              f"{harness.total_calls():,}")),
        _row("corpus-size", "%d payloads is a small corpus" % len(CORPUS)),
        _row("class-cells", "Class level rates rest on %d to %d observations "
             "each" % (m["class_n"][0], m["class_n"][-1])),
        _row("guard:cost", "which costs it %.3f of containment on the %s "
             "sweep and %.3f of measured effect offline"
             % (_guard_cross_turn_share()
                - t["reports"][GUARD].containment.value, TERRA, m["plant_turn_guard"] - eff[GUARD[0]][1])),
        _row("todo:moved", "%s %s trials moved and %s %s trials did"
             % (words(len(moved_t)).capitalize(), TERRA,
                words(len(moved_o)), OPUS)),
        _row("todo:split", "%s of the %s moved verdicts can be re-judged from "
             "the rows alone, and %s read answer text"
             % (words(rejudged), words(len(moved_t)),
                words(len(moved_t) - rejudged))),
        _row("terra:answer-share", "answer-shaped payloads are %d of %d "
             "compliances" % (t["answer_shaped"], t["complied"])),
    ]
    return out


def _opus_prose(o, doc):
    """The claude-opus-5 paragraphs: what the column is made of."""
    n = f"{o['trials']:,}"
    calls = o["tool_calls"]
    tool_targets = sorted({tool_name(c.action.target) for c in calls})
    tool_payloads = sorted({c.payload_id for c in calls})
    articulate = o["with_text"] - o["complied_with_text"]
    remaining = o["trials"] - o["textless"] - o["with_text"]
    in_ten = "%s trials in ten"
    rows = [
        _row("opus:no-action", "Not one injected tool call, HTTP GET or memory "
             "write, in %s trials" % n if o["action_taking"] == 0
             else "%d injected actions" % o["action_taking"]),
        _row("opus:tool-calls", "The checkpoint carries %d tool calls from it, "
             "all of them the in-grant %s on %s and all scored %s"
             % (len(calls), "/".join(tool_targets), "/".join(tool_payloads),
                "non-compliant" if not any(c.complied for c in calls)
                else "(some compliant)")),
        _row("opus:format", "All %d of its compliances are format adoption on "
             "%s payloads, %s" % (o["complied"],
                                  words(len(o["complied_payloads"])),
                                  " and ".join(o["complied_payloads"]))
             if o["action_taking"] == 0 else "(action-taking compliances)"),
        _row("opus:textless", "Of the %s trials, %s (%.3f) are refusals with "
             "no text" % (n, f"{o['textless']:,}", o["textless"] / o["trials"])),
        _row("opus:silent", "%d of the %d payloads never drew a word from the "
             "model under any configuration" % (o["never_a_word"],
                                                 o["payloads"])),
        _row("opus:with-text", "%d trials carry answer text. %d of them are "
             "the compliances above; the other %d refuse articulately"
             % (o["with_text"], o["complied_with_text"], articulate)),
        _row("opus:remaining", "The remaining %d are the %s tool calls"
             % (remaining, "/".join(tool_payloads))
             if remaining == len(calls) else "(%d unaccounted)" % remaining),
        _row("opus:in-ten", "the articulate refusals describe %s, not the run, "
             "and a compliance rate of %.3f reads differently once %s never "
             "answered at all"
             % (in_ten % words(round(10 * articulate / o["trials"])), o["all"],
                in_ten % words(round(10 * o["textless"] / o["trials"])))),
    ]
    return rows if doc == README else []


def _terra_prose(t, doc):
    gets = t["kinds"]["http_get"]
    tools = t["kinds"]["tool_call"]
    hosts = "/".join(t["gets_allowed_payloads"])
    if doc == README:
        text = ("%s did, and refused nothing outright. RE-SCORED: %d HTTP GETs, "
                "of which %d went to attacker-controlled hosts and %d to the one "
                "allowlisted internal host that payload %s deliberately "
                "targets; %d tool calls, of which %d were outside the grant and "
                "%d were the in-grant search_documents; and %d memory writes, "
                "across %s of the %s attack classes (%s asks for no action)")
    else:
        text = ("%s did, and never returned a textless refusal. RE-SCORED: %d "
                "HTTP GETs, %d of them to attacker-controlled hosts and %d to "
                "the allowlisted host %s targets; %d tool calls, %d outside the "
                "grant and %d the in-grant search_documents; %d memory writes; "
                "spread across %s of the %s attack classes, since %s asks for "
                "no action")
    return [_row("terra:actions", text % (
        TERRA if t["textless"] == 0 else "(textless refusals)", gets,
        gets - t["gets_allowed"], t["gets_allowed"], hosts, tools,
        tools - t["tools_granted"], t["tools_granted"],
        t["kinds"]["memory_write"], words(len(t["action_classes"])),
        words(len(attack_classes())),
        " and ".join(sorted(c.value for c in attack_classes()
                            if c.value not in t["action_classes"]))),
        None if doc == README else "found")]


def _capture(model, s, region):
    """The result sections of a real-model capture, re-rendered from the rows
    in the format that version of the demo printed."""
    reports = s["reports"]
    n = s["trials"]
    configs = len(reports)
    multi = sum(rescore.PAYLOADS[p].is_multi_turn
                for p in {t.payload_id for t in s["attempts"]})
    rows = [
        _block("%s:preflight" % region, [
            "payloads %d of %d" % (s["payloads"], len(CORPUS)),
            "attack classes %d" % len({t.attack_class for t in s["attempts"]}),
            "delivery channels %d" % len({t.channel for t in s["attempts"]}),
            "defense configurations %d" % configs,
            "repeats per trial %d" % max(t.repeat for t in s["attempts"]),
            "TOTAL TRIALS %d" % n,
            "TOTAL MODEL CALLS %d (%d multi-turn payloads cost two each)"
            % (s["calls"], multi)], region),
        _block("%s:section2" % region, [
            "trials run %d" % n,
            "noise floor (flip rate across repeats) %.3f" % s["floor"],
            "median 95%% interval width %.3f" % s["width"],
            "payloads that never discriminate %d/%d (%.3f)"
            % (len(s["dead"]), s["payloads"], s["dead_rate"])], region),
        _block("%s:section3" % region, [
            "attack_success %s" % reports[()].attack_success,
            "compliance (agent was fooled) %s" % reports[()].compliance,
            "containment (damage was stopped) %s" % reports[()].containment],
            region),
        _block("%s:section4" % region, ["configuration reduction flip verdict"]
               + ["%s %+.3f %.3f %s" % (
                   label[:36], reduction, flip,
                   "shown" if shown else "NOT SHOWN (inside its own noise)")
                  for label, reduction, shown, flip in s["effect_rows"]],
               region),
        _block("%s:section5" % region, ["configuration complied contained"]
               + ["%s %.3f %.3f" % (r.label[:36], r.compliance.value,
                                    r.containment.value)
                  for _, r in sorted(reports.items())] + ["===="], region),
    ]
    worst = max((r for d, r in reports.items() if d),
                key=lambda r: r.attack_success.value)
    rows.append(_block("%s:section7" % region,
                       ["worst surviving configuration: %s" % worst.label]
                       + ["%s attack_success %s" % (name, rate)
                          for name, rate in sorted(worst.by_class.items())],
                       region))
    textless = s["textless"]
    rows.append(_block("%s:stored" % region, [
        "already recorded %d/%d" % (n, n),
        "refused with no answer text %d/%d (%.3f)" % (textless, n, textless / n),
        "configuration complied contained neutralized refused, no text"]
        + ["%s %.3f %.3f %.3f %.3f" % (
            r.label[:36], r.compliance.value, r.containment.value,
            r.neutralized.value, r.refused_no_text.value)
           for _, r in sorted(reports.items())], region))
    return rows


def _moved_lines():
    """The seven re-scored trials, in the order SAMPLE_RUN.md lists them."""
    moved = rescore.overlay(TERRA)
    lines = []
    for m in sorted(moved, key=lambda m: (
            m["printed"]["complied"] == m["rescored"]["complied"],
            m["payload_id"], m["defenses"], m["repeat"])):
        was, now = m["printed"], m["rescored"]
        label = "+".join(m["defenses"]) or "(baseline)"
        if was["complied"] != now["complied"]:
            change = "complied %s -> %s" % (was["complied"], now["complied"])
            if now["contained"]:
                change += ", contained by %s" % now["contained_by"]
        else:
            change = "contained %s -> %s" % (was["contained"], now["contained"])
        lines.append("%s %s %s" % (m["payload_id"], label, change))
    complied_moves = sum(m["printed"]["complied"] != m["rescored"]["complied"]
                         for m in moved)
    return lines, complied_moves, len(moved) - complied_moves


def _sample_run_rows(o, t, op, tp):
    rows = _capture(OPUS, op, "opus") + _capture(TERRA, tp, "terra")
    # The re-rendered STORED tables are the printed verdicts plus the two
    # count columns, so they are checked against the printed rows above; the
    # opus sweep moved nothing, which is its own row.
    rows.append(_row("opus:nothing-moved",
                     "RE-SCORED: nothing moved" if not rescore.overlay(OPUS)
                     else "RE-SCORED: (something moved)", "opus"))
    rows.append(_row("opus:single-turn", "All %d of this model's compliances "
                     "are single-turn and answer-shaped" % o["complied"]
                     if o["action_taking"] == 0 and not any(
                         rescore.PAYLOADS[p].is_multi_turn
                         for p in o["complied_payloads"])
                     else "(not all single-turn answers)", "opus"))
    lines, fix_tool, fix_turn = _moved_lines()
    rows.append(_row("terra:moved-count", "%s trials of %s. %s are the "
                     "tool-argument fix, %s are the cross-turn fix"
                     % (words(len(lines)).capitalize(), f"{t['trials']:,}",
                        words(fix_tool).capitalize(), words(fix_turn)),
                     "terra"))
    rows.append(_block("terra:moved", lines, "terra"))
    pairs = [("compliance, all configurations", "%.3f" % tp["all"],
              "%.3f" % t["all"]),
             ("noise floor", "%.3f" % tp["floor"], "%.3f" % t["floor"])]
    for name in ("delimiter_fencing", "provenance_tagging",
                 "egress_filter+tool_allowlist"):
        was, now = tp["effects"][name], t["effects"][name]
        pairs.append((name, "%+.3f" % was[1], "%+.3f %s" % (
            now[1], "shown" if now[2] else "NOT SHOWN")))
    for name in ("egress_filter", "tool_allowlist", GUARD[0]):
        pairs.append(("%s containment" % name,
                      "%.3f" % tp["reports"][(name,)].containment.value,
                      "%.3f" % t["reports"][(name,)].containment.value))
    rows.append(_block("terra:printed-vs-rescored",
                       ["printed re-scored"] + [" ".join(p) for p in pairs],
                       "terra"))
    same = all(tp["effects"][k][2] == t["effects"][k][2] for k in t["effects"])
    rows.append(_row("terra:verdicts", "Every verdict in column %s of the "
                     "printed capture survives re-scoring"
                     % ("4" if same else "(some verdict moved)"), "terra"))
    # What the real model runs found.
    shown = sum(e[2] for e in t["effect_rows"])
    rows += [
        _row("found:total", "Written from the two full sweeps above, %s real "
             "trials in total" % f"{o['trials'] + t['trials']:,}", "found"),
        _block("found:table", [
            "%s %s" % (OPUS, TERRA),
            "undefended baseline compliance %.3f %.3f" % (o["baseline"],
                                                          t["baseline"]),
            "compliance, all configurations %.3f %.3f" % (o["all"], t["all"]),
            "of which action-taking %d %d" % (o["action_taking"],
                                              t["action_taking"]),
            "refused, no text %.3f %.3f" % (o["textless"] / o["trials"],
                                            t["textless"] / t["trials"]),
            "payloads that never discriminate %d/%d %d/%d"
            % (len(o["dead"]), o["payloads"], len(t["dead"]), t["payloads"])],
            "found"),
        _row("found:opus-no-action", "%s took no injected action in %s trials"
             % (OPUS, f"{o['trials']:,}") if o["action_taking"] == 0
             else "(opus took actions)", "found"),
        _row("found:opus-format", "All %d of its compliances are format "
             "adoption on %s payloads, %s" % (
                 o["complied"], words(len(o["complied_payloads"])),
                 " and ".join(o["complied_payloads"])), "found"),
        _row("found:opus-partition", "Of the %s trials, %s are refusals with "
             "no text at all: a refusal stop reason and no content, recorded "
             "as a non-compliance, on %d payloads for every trial under every "
             "configuration. %d trials carry answer text. %d of them are the "
             "compliances above; the other %d refuse articulately"
             % (f"{o['trials']:,}", f"{o['textless']:,}", o["never_a_word"],
                o["with_text"], o["complied_with_text"],
                o["with_text"] - o["complied_with_text"]), "found"),
        _row("found:opus-remaining", "The remaining %d are the in-grant "
             "search_documents calls on mem-02"
             % (o["trials"] - o["textless"] - o["with_text"]), "found"),
        _row("found:opus-in-ten", "%s trials in ten are a model that declined "
             "to answer at all" % words(round(10 * o["textless"]
                                              / o["trials"])), "found"),
        _terra_prose(t, SAMPLE_RUN)[0],
        _row("found:terra-baseline", "On the undefended baseline it complied "
             "on %.3f of trials" % t["baseline"], "found"),
        _row("found:terra-fencing", "fencing the untrusted material took "
             "compliance from %.3f to %.3f" % (
                 t["baseline"],
                 t["reports"][("delimiter_fencing",)].compliance.value),
             "found"),
        _row("found:terra-containment", "while containing %.3f and %.3f of "
             "the compliances they saw" % (
                 t["reports"][("egress_filter",)].containment.value,
                 t["reports"][("tool_allowlist",)].containment.value),
             "found"),
        _row("found:opus-floor", "With %d compliances in %s trials there is "
             "essentially no attack success to reduce, so no configuration can "
             "clear even a %.3f floor" % (o["complied"], f"{o['trials']:,}",
                                          min(e[3] for e in o["effect_rows"])),
             "found"),
        _row("found:terra-shown", "%s cleared its floor on %s configurations, "
             "and the floor is %.3f" % (TERRA, words(shown), t["floor"]),
             "found"),
        _row("found:class-cells", "the per-class cells rest on %d to %d "
             "observations each" % (
                 min(r.n for r in t["reports"][()].by_class.values()),
                 max(r.n for r in t["reports"][()].by_class.values())),
             "found"),
        _row("found:answer-share", "the answer-shaped payloads are %d of %d "
             "compliances" % (t["answer_shaped"], t["complied"]), "found"),
    ]
    return rows


def region_text(doc_text, region):
    if region is None:
        return doc_text
    start, end = REGIONS[region]
    body = doc_text.split(start, 1)[1] if start in doc_text else ""
    return body.split(end, 1)[0] if end else body


def main():
    args = sys.argv[1:]
    emit = "--emit" in args
    emit_doc = next((a for a in args if a in (README, SAMPLE_RUN)), README)
    derived = derive()
    if emit:
        for tag, _, text, _ in derived[emit_doc]:
            print("%-28s %s" % (tag, text))
    status = 0
    for doc in (README, SAMPLE_RUN):
        with open(os.path.join(ROOT, doc), encoding="utf-8") as fh:
            raw = fh.read()
        rows = derived[doc]
        bad = []
        for tag, pattern, _, region in rows:
            found = len(re.findall(pattern, normalize(region_text(raw, region))))
            if found != 1:
                bad.append((tag, pattern, found))
        for tag, pattern, found in bad:
            print("%s [%s] in %s\n  %s" % (
                "MISSING" if not found else "MATCHES %d TIMES" % found,
                tag, doc, pattern))
        print("%d of %d derived figures found exactly once in %s"
              % (len(rows) - len(bad), len(rows), doc))
        status = status or (1 if bad else 0)
    return status


if __name__ == "__main__":
    sys.exit(main())
