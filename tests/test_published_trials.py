"""The published per-trial rows must reproduce the published real-model figures.

A README that says its figures are reproducible, in a repository whose subject
is measurement, has to be able to fail. This is that test: it reads only the
files under `results/`, the ones a reader actually gets, and asserts they
carry every field the scoring consumes, and that scoring them gives the same
answer as scoring the unshipped checkpoint.

It is skipped when the checkpoint is absent, because the comparison needs both.
The first assertion does not need the checkpoint and never skips.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PUBLISHED = {
    ROOT / "results/trials.claude-opus-5.jsonl": ROOT / "audit/full.jsonl",
    ROOT / "results/trials.gpt-5_6-terra.jsonl": ROOT / "audit/full.openai.jsonl",
}
SCORED = ("payload_id", "defenses", "repeat", "complied", "contained",
          "contained_by", "neutralized_by", "attack_class", "channel")


def _rows(path: pathlib.Path) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines][1:]


@pytest.mark.parametrize("published", sorted(PUBLISHED), ids=lambda p: p.name)
def test_every_scored_field_is_present_and_no_prose_is(published: pathlib.Path) -> None:
    rows = _rows(published)
    assert rows, f"{published.name} carries no trials"
    for row in rows:
        for field in SCORED:
            assert field in row, f"{published.name} drops {field}, which the scoring reads"
        action = row["action"]
        assert set(action) == {"kind", "target", "answer_len", "provoked_by_len"}, (
            f"{published.name} action carries {sorted(action)}; publishing the "
            "model's prose is the thing this file exists to avoid"
        )
        for length in ("answer_len", "provoked_by_len"):
            assert isinstance(action[length], int)


@pytest.mark.parametrize("published,checkpoint", sorted(PUBLISHED.items()),
                         ids=lambda p: getattr(p, "name", ""))
def test_the_published_rows_score_identically_to_the_checkpoint(
    published: pathlib.Path, checkpoint: pathlib.Path
) -> None:
    if not checkpoint.exists():
        pytest.skip(f"{checkpoint.name} is not present; nothing to compare against")
    theirs, ours = _rows(checkpoint), _rows(published)
    assert len(theirs) == len(ours)
    for a, b in zip(theirs, ours):
        for field in SCORED:
            assert a.get(field) == b.get(field), (
                f"{published.name} disagrees with the checkpoint on {field}: "
                f"{a.get(field)!r} against {b.get(field)!r}"
            )
        # The action's kind and target are published and the scoring reads
        # them, so they are compared too; the text lengths stand in for text.
        for field in ("kind", "target"):
            assert a["action"][field] == b["action"][field], (
                f"{published.name} disagrees with the checkpoint on "
                f"action.{field}: {a['action'][field]!r} against "
                f"{b['action'][field]!r}"
            )
        assert b["action"]["answer_len"] == len(a["action"]["answer"])
        assert b["action"]["provoked_by_len"] == len(a["action"]["provoked_by"])


# --- the RE-SCORED figures, from what a reader gets ------------------------
#
# The rows carry the verdicts each sweep PRINTED. results/rescored.*.json
# carries the verdicts two later scoring fixes moved, and the README's
# RE-SCORED tables are the rows with that file applied. Neither test below
# needs the unshipped checkpoint, so neither skips in CI.

import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))

import rescore  # noqa: E402
from bench.harness import judge  # noqa: E402


@pytest.mark.parametrize("model", sorted(rescore.SOURCES))
def test_every_verdict_that_reads_no_answer_text_rejudges_from_the_row(model):
    """judge() over the published row gives the RE-SCORED verdict, for every
    trial whose verdict reads only the recorded action. That covers the moved
    trials the rows can check and every trial that did NOT move, so the
    overlay cannot leave out a verdict the shipped scoring would change."""
    moved = {(m["payload_id"], tuple(m["defenses"]), m["repeat"]): m
             for m in rescore.overlay(model)}
    checked = 0
    for row in rescore.rows(model):
        if rescore.needs_answer_text(row["payload_id"], row["defenses"]):
            continue
        want = moved.get(rescore.key(row), {}).get(
            "rescored", {f: row[f] for f in rescore.VERDICT})
        got = judge(rescore.PAYLOADS[row["payload_id"]],
                    frozenset(row["defenses"]), rescore._action(row["action"]))
        assert got == tuple(want[f] for f in rescore.VERDICT), (
            f"{model} {rescore.key(row)}: judged {got}, published {want}")
        checked += 1
    assert checked > 1000, f"only {checked} rows could be re-judged"


def test_the_overlay_marks_what_the_rows_cannot_check():
    for model in rescore.SOURCES:
        for m in rescore.overlay(model):
            assert m["needs_answer_text"] == rescore.needs_answer_text(
                m["payload_id"], m["defenses"])


def test_the_readme_checker_passes_on_the_shipped_readme():
    import check_readme_numbers as chk
    assert chk.main() == 0


def _check_with_one_edit(doc, shipped, edited, monkeypatch, tmp_path):
    """Run the checker against copies of both documents, one of them edited."""
    import check_readme_numbers as chk
    for name in (chk.README, chk.SAMPLE_RUN):
        text = (ROOT / name).read_text(encoding="utf-8")
        if name == doc:
            assert text.count(shipped) == 1, shipped
            text = text.replace(shipped, edited)
        (tmp_path / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(chk, "ROOT", str(tmp_path))
    return chk.main()


@pytest.mark.parametrize("shipped, edited", [
    ("that\nmodel complied 27 times", "that\nmodel complied 31 times"),
    ("refused, no text                    0.699",
     "refused, no text                    0.599"),
    ("undefended baseline compliance      0.026              0.365",
     "undefended baseline compliance      0.126              0.465"),
    ("delimiter_fencing            NOT SHOWN   NOT SHOWN   +0.333 shown",
     "delimiter_fencing            NOT SHOWN   NOT SHOWN   +0.433 shown"),
    # Real-model figures stated in prose, one per sentence.
    ("printed\n486,", "printed\n386,"),
    ("the run printed 409)", "the run printed 309)"),
    ("15 tool calls from it", "25 tool calls from it"),
    ("and 23 of the 52 payloads", "and 33 of the 52 payloads"),
    ("the other 568 refuse", "the other 595 refuse"),
    ("which 188 went", "which 288 went"),
    ("and 3 to the one allowlisted", "and 13 to the one allowlisted"),
    ("which 152 were outside", "which 252 were outside"),
    ("and 8 were the in-grant", "and 18 were the in-grant"),
    ("; four do on", "; five do on"),
    ("0.038 floor", "0.138 floor"),
    ("+0.141 PRINTED", "+0.241 PRINTED"),
    ("floor of 0.173", "floor of 0.273"),
    ("from 0.365 to 0.032", "from 0.365 to 0.132"),
    ("contained 0.373", "contained 0.473"),
    ("contained 0.286", "contained 0.386"),
    ("-0.019 RE-SCORED", "-0.119 RE-SCORED"),
    ("+0.006 on", "+0.106 on"),
    ("costs it 0.055", "costs it 0.155"),
    ("0.019 of measured effect", "0.119 of measured effect"),
    # Offline figures stated a second time, outside the twelve-row table.
    ("egress_filter                  +0.327", "egress_filter                  +0.427"),
    ("`tool_allowlist` buys +0.269", "`tool_allowlist` buys +0.369"),
    ("Fencing measures +0.269", "Fencing measures +0.369"),
    ("flip rate is 0.654", "flip rate is 0.754"),
    ("It measures +0.154.", "It measures +0.254."),
])
def test_one_edited_figure_fails_the_readme_checker(shipped, edited,
                                                    monkeypatch, tmp_path):
    assert _check_with_one_edit("README.md", shipped, edited,
                                monkeypatch, tmp_path) == 1


@pytest.mark.parametrize("shipped, edited", [
    # The RE-SCORED table, the printed-against-re-scored table and the prose.
    ("compliance, all configurations       0.013            0.242",
     "compliance, all configurations       0.013            0.342"),
    ("output_provenance_guard containment     0.167        0.109",
     "output_provenance_guard containment     0.167        0.209"),
    ("cleared its floor on five configurations",
     "cleared its floor on six configurations"),
    ("595 trials carry\nanswer text. 27 of them", "595 trials carry\nanswer text. 17 of them"),
    # A printed capture and a re-rendered STORED table.
    ("attack_success                         0.359 [0.282, 0.429] n=156",
     "attack_success                         0.459 [0.282, 0.429] n=156"),
    ("  unicode_normalization                     0.019       0.000        0.000             0.788",
     "  unicode_normalization                     0.019       0.000        0.000             0.688"),
    # One of the seven moved trials.
    ("contained by tool_allowlist", "contained by egress_filter"),
])
def test_one_edited_figure_fails_the_sample_run_check(shipped, edited,
                                                      monkeypatch, tmp_path):
    assert _check_with_one_edit("SAMPLE_RUN.md", shipped, edited,
                                monkeypatch, tmp_path) == 1


def test_a_figure_stated_twice_fails_the_readme_checker(monkeypatch, tmp_path):
    """A row must match exactly once: with two copies of a sentence, one of
    them could be edited and the row would still find the other."""
    assert _check_with_one_edit("README.md", "It measures +0.154.",
                                "It measures +0.154. It measures +0.154.",
                                monkeypatch, tmp_path) == 1
