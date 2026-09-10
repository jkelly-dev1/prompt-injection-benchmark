"""The demo prints the two columns that a compliance figure cannot be read without.

`neutralized_by` and a textless refusal are both recorded on every Attempt and
both are computed by the scoring layer, and a field that is computed and never
printed is a field a reader cannot use. So the report is driven end to end on
a small offline matrix and the printed text is asserted, not the objects
behind it.

The demo is imported from its file rather than shelled out to, so a failure
is a Python traceback and not an exit code, and the working directory is a
temporary one because the demo writes an audit record wherever it is run.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_demo_main():
    spec = importlib.util.spec_from_file_location(
        "run_demo", REPO_ROOT / "scripts" / "run_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.main


def _row(output: str, label: str) -> list[str]:
    """The section-5 row for a configuration label, split into its cells."""
    for line in output.splitlines():
        cells = line.split()
        if cells and cells[0] == label and len(cells) >= 5:
            return cells
    raise AssertionError(f"no section-5 row for {label!r} in the demo output")


def test_the_demo_prints_neutralized_and_textless_refusal_beside_compliance(
    tmp_path, monkeypatch, capsys
):
    """Both columns are in the per-configuration table, and the refusal total is
    in the run header.

    Three payloads, one from each of the first three attack classes, under the
    reduced configuration set, one repeat: 18 trials. The mock never refuses,
    so the refusal figures are exactly zero everywhere, and the filter does
    neutralize at least one of the three payloads, so the neutralized column is
    non-zero on the filter's row and zero on the undefended one.

    Mutation checks: delete either header cell and the first assertions fail;
    count textless answers instead of marked refusals and the mock's own
    no-directive answers make the total non-zero.
    """
    monkeypatch.chdir(tmp_path)
    main = _run_demo_main()
    assert main(["--payloads", "3", "--reduced-configs", "--repeats", "1"]) == 0
    output = capsys.readouterr().out

    header = next(
        line for line in output.splitlines()
        if line.strip().startswith("configuration") and "complied" in line
    )
    assert "neutralized" in header
    assert "refused, no text" in header
    assert re.search(r"refused with no answer text\s+0/18 \(0\.000\)", output), output

    undefended = _row(output, "(none)")
    assert undefended[-2:] == ["0.000", "0.000"], undefended

    filtered = _row(output, "input_pattern_filter")
    neutralized, refused = float(filtered[-2]), filtered[-1]
    assert 0.0 < neutralized <= 1.0, filtered
    assert refused == "0.000", filtered
