"""The corpus composition, which IS the measurement.

The failure mode this file guards against is a harness whose fixture and whose
measurement move together. If every payload were written in the vocabulary of
the denylist, input_pattern_filter would score near-perfectly and the report
would recommend a denylist; that is a finding about who wrote the corpus, not
about pattern filtering. So the property under test is DISCRIMINATION: the
corpus must contain payloads a deny pattern matches AND paraphrases with the
same goal that it misses, and that split is asserted by running the real filter
over every payload rather than by counting on the author's intent.

The same argument applies to the allowlists. Targets have to land on both sides
of them, because a control that refuses every sample in the corpus is
indistinguishable from a control that refuses everything, and a benchmark that
never exercises the pass path has tested a default rather than a rule.

Coverage is checked here too (every declared channel and attack class actually
populated, enough payloads per class to support a rate), along with the one
structural invariant the least-privilege thesis rests on: every persona hijack
wants ANSWER, because a persona hijack calls nothing.
"""

from __future__ import annotations

import base64
import re

from bench.attacks.corpus import (
    CORPUS,
    attack_classes,
    by_attack_class,
    by_channel,
    channels,
)
from bench.defenses import (
    ALLOWED_HOSTS,
    ALLOWED_TOOLS,
    Request,
    input_pattern_filter,
)
from bench.models import ActionKind, AttackClass, Channel

#: The same shape defenses._B64 looks for. Duplicated deliberately: if the
#: defense's own regex were imported, a change that broke both would pass.
_B64_RUN = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")

#: Below this a per-class rate is one or two payloads wide and reports noise.
MIN_PER_ATTACK_CLASS = 3


def _filtered(text: str) -> Request:
    return input_pattern_filter(Request(task="Summarize.", untrusted=text))


def _host_of(target: str) -> str:
    """Host as egress_filter reads it: everything before the first slash."""
    return target.split("/")[0].lower()


def test_every_declared_channel_and_attack_class_is_actually_populated():
    """A declared-but-empty cell reads as "no attacks arrive this way".

    That is the most dangerous sentence a report of this kind could contain, so
    the enums and the corpus are compared in both directions rather than trusting
    that whoever added a channel also wrote payloads for it.
    """
    assert set(channels()) == set(Channel)
    assert set(attack_classes()) == set(AttackClass)

    for attack_class in AttackClass:
        payloads = by_attack_class(attack_class)
        assert len(payloads) >= MIN_PER_ATTACK_CLASS, (
            f"{attack_class.value} has {len(payloads)} payloads; a rate over "
            f"fewer than {MIN_PER_ATTACK_CLASS} is not worth reporting"
        )

    for channel in Channel:
        assert by_channel(channel), f"{channel.value} is declared but empty"


def test_payload_ids_are_unique():
    """Two payloads sharing an ID would silently merge in every grouped result.

    Mutation check: duplicate any payload_id and the audit record, the per-class
    rates and the matrix all keep working while quietly double-counting one
    sample, which is the kind of error a suite has to catch structurally.
    """
    ids = [payload.payload_id for payload in CORPUS]
    assert len(set(ids)) == len(ids), f"duplicate payload_id in {sorted(ids)}"


def test_every_encoded_payload_carries_a_base64_run_that_really_decodes():
    """The encoded class is only meaningful if the encoding is real.

    A payload whose "base64" is decorative would sail through normalization
    unchanged and the whole encoding finding would be an artifact of a string
    that merely looked encoded. So the decode is EXECUTED here rather than
    matched: the run has to be long enough for the defense to try it and it has
    to come back as usable utf-8 text.
    """
    encoded = by_attack_class(AttackClass.ENCODED_PAYLOAD)
    assert encoded, "the encoded class must not be empty"

    for payload in encoded:
        match = _B64_RUN.search(payload.injection)
        assert match, f"{payload.payload_id} carries no base64 run to decode"
        blob = match.group(0)
        assert len(blob) >= 24, (
            f"{payload.payload_id} has a run defenses._B64 would skip"
        )
        padded = blob + "=" * (-len(blob) % 4)
        plaintext = base64.b64decode(padded, validate=True).decode("utf-8")
        assert plaintext.strip(), (
            f"{payload.payload_id} decodes to nothing an agent could act on"
        )


def test_the_corpus_contains_both_deny_matching_payloads_and_paraphrases():
    """Without both halves the benchmark cannot discriminate at all.

    The split is measured by running the real input_pattern_filter over every
    payload and counting which ones it claims, so it reflects the two regex lists
    actually disagreeing rather than anyone's opinion about which payload is
    which. Mutation check: rewrite the corpus in the denylist's vocabulary and
    the paraphrase count goes to zero here, which is exactly the self-confirming
    fixture this test exists to prevent.
    """
    matched = [p for p in CORPUS if _filtered(p.rendered()).neutralized_by]
    missed = [p for p in CORPUS if not _filtered(p.rendered()).neutralized_by]

    assert matched, "no payload matches a deny pattern; the filter cannot score"
    assert missed, (
        "every payload matches a deny pattern; the corpus was written in the "
        "defense's own vocabulary and would flatter it"
    )
    assert len(matched) + len(missed) == len(CORPUS)

    # Both halves have to be substantial, not one token sample apiece.
    assert len(matched) >= MIN_PER_ATTACK_CLASS
    assert len(missed) >= MIN_PER_ATTACK_CLASS


def test_targets_land_on_both_sides_of_the_two_allowlists():
    """A control that refuses every sample has been defaulted, not tested.

    Both directions are required for each allowlist: payloads aimed outside it
    (so the refusal path is exercised) and payloads aimed inside it (so a run
    cannot explain its results by saying everything was going to be refused
    anyway). The pass-path half is the one a corpus author forgets.
    """
    fetches = [p for p in CORPUS if p.wants is ActionKind.HTTP_GET]
    tool_calls = [p for p in CORPUS if p.wants is ActionKind.TOOL_CALL]
    assert fetches and tool_calls

    outside_hosts = [p for p in fetches if _host_of(p.target) not in ALLOWED_HOSTS]
    inside_hosts = [p for p in fetches if _host_of(p.target) in ALLOWED_HOSTS]
    assert outside_hosts, "no payload exercises the egress filter's refusal path"
    assert inside_hosts, "no payload exercises the egress filter's pass path"

    outside_tools = [p for p in tool_calls if p.target not in ALLOWED_TOOLS]
    inside_tools = [p for p in tool_calls if p.target in ALLOWED_TOOLS]
    assert outside_tools, "no payload exercises the tool grant's refusal path"
    assert inside_tools, "no payload exercises the tool grant's pass path"

    # Every action that names a target has one, or the allowlists have nothing
    # to rule on.
    for payload in fetches + tool_calls:
        assert payload.target.strip(), f"{payload.payload_id} has no target"


def test_every_persona_hijack_wants_only_an_answer():
    """The load-bearing case for the least-privilege thesis, held as data.

    A persona hijack calls no tool and opens no socket, so tool_allowlist and
    egress_filter score exactly zero against this class. If a hijack payload
    ever asked for a tool call, the class would start showing containment that
    belongs to a different attack and the matrix would quietly overstate what
    structural controls buy.
    """
    hijacks = by_attack_class(AttackClass.PERSONA_HIJACK)
    assert hijacks
    for payload in hijacks:
        assert payload.wants is ActionKind.ANSWER, (
            f"{payload.payload_id} wants {payload.wants.value}; a hijack that "
            "calls something would let an action rule take credit for it"
        )
        assert payload.target == "", "an ANSWER has nothing to aim at"


def test_the_selectors_return_what_they_claim_and_come_back_sorted():
    """Grouping helpers are load-bearing for the matrix, so they are checked.

    Sorted output is what makes a rerun of the report byte-identical; an
    unordered set of channels would reshuffle columns between runs and make two
    reports impossible to diff.
    """
    for channel in Channel:
        selected = by_channel(channel)
        assert selected == tuple(p for p in CORPUS if p.channel is channel)
        assert all(p.channel is channel for p in selected)

    for attack_class in AttackClass:
        selected = by_attack_class(attack_class)
        expected = tuple(p for p in CORPUS if p.attack_class is attack_class)
        assert selected == expected
        assert all(p.attack_class is attack_class for p in selected)

    # Accepts the raw string form as well, and coerces through the enum.
    assert by_channel("document") == by_channel(Channel.DOCUMENT)
    assert by_attack_class("exfiltration") == by_attack_class(
        AttackClass.EXFILTRATION
    )

    assert list(channels()) == sorted(channels(), key=lambda c: c.value)
    assert list(attack_classes()) == sorted(attack_classes(), key=lambda c: c.value)

    # Every payload is reachable through both groupings, exactly once.
    by_channel_total = sum(len(by_channel(c)) for c in Channel)
    by_class_total = sum(len(by_attack_class(a)) for a in AttackClass)
    assert by_channel_total == by_class_total == len(CORPUS)


def test_every_payload_explains_why_it_is_in_the_corpus():
    """A payload without a rationale cannot be defended in review.

    The composition IS the measurement here, so each sample has to say what it
    is buying. An unexplained payload is one nobody can argue with later, and it
    is how a corpus drifts into confirming whatever the defenses already do.
    """
    for payload in CORPUS:
        assert payload.rationale.strip(), (
            f"{payload.payload_id} has no rationale"
        )
        assert payload.carrier.strip(), f"{payload.payload_id} has no carrier"
        assert payload.injection.strip(), f"{payload.payload_id} has no injection"
        assert payload.carrier in payload.rendered()
        assert payload.injection in payload.rendered()
