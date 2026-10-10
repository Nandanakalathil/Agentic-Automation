"""The orientation document Agent 1 produces.

The diagram tests matter most: the analyst supplies nodes and edges, never
diagram syntax, so a malformed step must still render a valid chart rather than
an error box on the page.
"""
import pytest

from pipeline import reports

STATE = {"run_id": "r1", "repo_url": "https://github.com/o/r", "commit_sha": "abc", "run_dir": "/tmp"}


def render(**over) -> str:
    a = {"summary": "s", "task": "t", "preflight": {"tree": ["./  (2 files)"]}}
    a.update(over)
    return reports.repo_analysis(STATE, a, "narrative")


# ------------------------------------------------------------------ diagram
def test_flow_renders_shapes_and_edges():
    out = reports.flow_diagram([
        {"id": "a", "label": "In", "kind": "input", "next": ["b"]},
        {"id": "b", "label": "Model", "kind": "model", "next": ["c"]},
        {"id": "c", "label": "Out", "kind": "output"},
    ])
    assert out.startswith("```mermaid\nflowchart TD")
    assert 'a(["In"]):::input' in out
    assert 'b[["Model"]]:::model' in out
    assert "a --> b" in out and "b --> c" in out


def test_edge_to_an_unknown_step_is_dropped():
    """A dangling edge makes Mermaid render an error box instead of the chart."""
    out = reports.flow_diagram([{"id": "a", "label": "A", "next": ["ghost"]}])
    assert "ghost" not in out


def test_ids_with_spaces_and_punctuation_are_sanitised():
    out = reports.flow_diagram([
        {"id": "msa search!", "label": "MSA search", "kind": "external", "next": ["fold-step"]},
        {"id": "fold-step", "label": 'Fold "trunk"', "kind": "model"},
    ])
    assert "msa_search" in out and "msa search!" not in out
    assert "fold_step --" not in out  # no stray edge syntax
    assert '"Fold \'trunk\'"' in out  # inner quotes neutralised, not left raw


def test_unknown_kind_falls_back_to_process():
    out = reports.flow_diagram([{"id": "x", "label": "X", "kind": "nonsense"}])
    assert 'x["X"]:::process' in out


def test_missing_id_uses_the_label():
    assert "Step_one" in reports.flow_diagram([{"label": "Step one"}])


def test_empty_flow_renders_nothing():
    assert reports.flow_diagram([]) == ""
    assert "## Process flow" not in render()


def test_long_labels_are_clipped():
    out = reports.flow_diagram([{"id": "a", "label": "z" * 500}])
    node = next(l for l in out.splitlines() if l.strip().startswith("a["))
    assert len(node) < 120


# ------------------------------------------------------------------- alerts
def test_blockers_sort_above_warnings_and_info():
    out = render(alerts=[{"severity": "info", "title": "minor"},
                         {"severity": "blocker", "title": "serious"},
                         {"severity": "warning", "title": "middling"}])
    assert out.index("serious") < out.index("middling") < out.index("minor")


def test_machine_findings_are_separated_from_analyst_judgement():
    out = render(alerts=[{"severity": "warning", "title": "model thinks so"}],
                 risks=["LICENCE BLOCKER (weights): non-commercial"])
    assert out.index("Verified by direct file inspection") < out.index("Reported by the analyst")
    assert out.index("non-commercial") < out.index("model thinks so")


def test_agent_freeform_risks_do_not_reach_the_banner():
    """Only the prefixed, machine-checked findings are presented as fact."""
    out = render(risks=["the code is a bit messy"])
    assert "the code is a bit messy" not in out.split("## What this is")[0]


def test_clean_repo_says_so():
    assert "No blocking issues found" in render()
    assert "\u2705" not in render()


def test_banner_precedes_everything():
    out = render(alerts=[{"severity": "blocker", "title": "stop"}])
    assert out.index("Read this first") < out.index("## Metadata")


# ----------------------------------------------------------------- metadata
def test_optional_inputs_excluded_from_the_metadata_input_row():
    out = render(inputs=[{"name": "seq", "format": "fasta", "required": True},
                         {"name": "temp", "format": "float", "required": False}])
    row = next(l for l in out.splitlines() if l.startswith("| **Input**"))
    assert "fasta" in row and "float" not in row


def test_outputs_table_has_no_required_column():
    out = render(outputs=[{"name": "pdb", "format": "pdb"}])
    section = out.split("## Outputs")[1].split("##")[0]
    assert "Required" not in section


def test_scanned_links_fill_in_when_the_analyst_missed_them():
    out = render(preflight={"tree": [], "links": {"arxiv": ["https://arxiv.org/abs/1234.5678"]}})
    assert "1234.5678" in out


def test_peak_memory_shows_its_source():
    out = render(hardware={"peak_memory_gb": 23.5, "peak_memory_basis": "README"})
    assert "23.5 GB (source: README)" in out


def test_asset_availability_distinguishes_bundled_from_download():
    out = render(assets=[{"path": "w.pt", "kind": "weights", "bundled": False},
                         {"path": "c.yaml", "kind": "config", "bundled": True}])
    assert "download" in out and "in repo" in out


# -------------------------------------------------------------- robustness
def test_renders_from_the_thinnest_possible_analysis():
    """RepoAnalysis requires only summary and task; the report must still build."""
    out = reports.repo_analysis(STATE, {"summary": "s", "task": "t"}, "")
    assert "## Metadata" in out and "## Inputs" in out


@pytest.mark.parametrize("field", ["capabilities", "scripts", "assets", "inputs", "outputs",
                                   "flow", "alerts", "key_files", "functions"])
def test_null_instead_of_list_does_not_crash(field):
    """LLM-repaired JSON carries nulls where lists were expected."""
    assert reports.repo_analysis(STATE, {"summary": "s", "task": "t", field: None}, "")


def test_pipe_characters_in_values_do_not_break_tables():
    out = render(scripts=[{"path": "a.py", "kind": "cli", "invocation": "a | b | c"}])
    assert r"a \| b \| c" in out


# --------------------------------------------------------------- quickstart
def test_quickstart_renders_above_metadata():
    out = render(quickstart={"install": ["pip install -e ."], "run": "python fold.py -i x.fa",
                             "expected_output": "writes out/x.pdb"})
    assert "## Quickstart" in out
    assert out.index("## Quickstart") < out.index("## Metadata")
    assert "pip install -e ." in out and "writes out/x.pdb" in out


def test_quickstart_omitted_when_no_command_was_established():
    """A wrong command is worse than none, so an empty quickstart renders nothing."""
    assert "## Quickstart" not in render(quickstart={"caveats": ["needs a GPU"]})
    assert "## Quickstart" not in render()


def test_quickstart_caveats_shown():
    out = render(quickstart={"run": "python a.py", "caveats": ["needs 24 GB VRAM"]})
    assert "Watch out for" in out and "needs 24 GB VRAM" in out


# ------------------------------------------------------------ broken scripts
def test_broken_scripts_appear_in_the_verified_banner():
    out = render(risks=["BROKEN SCRIPT: 2 Python file(s) do not parse: old.py (line 1: ...)"])
    assert "Verified by direct file inspection" in out
    assert "BROKEN SCRIPT" in out.split("## What this is")[0]


# ----------------------------------------------------------------- fallbacks
def test_status_is_annotated_with_commit_age():
    out = render(metadata={"status": "active"},
                 preflight={"tree": [], "days_since_commit": 900})
    assert "active (last commit 2.5 years ago)" in out


def test_model_size_falls_back_to_bundled_weights():
    out = render(preflight={"tree": [], "bundled_weights_mb": 2600.0})
    assert "2.6 GB bundled in repo" in out


# ------------------------------------------------------ no emojis, no repeats
def test_report_contains_no_emoji():
    out = render(alerts=[{"severity": "blocker", "title": "x"},
                         {"severity": "warning", "title": "y"},
                         {"severity": "info", "title": "z"}],
                 risks=["LICENCE BLOCKER (weights): non-commercial"])
    for ch in ("\U0001F6D1", "⚠", "ℹ", "✅", "█", "░"):
        assert ch not in out, f"emoji {ch!r} still rendered"


def test_severity_still_readable_without_icons():
    out = render(alerts=[{"severity": "blocker", "title": "stop"}])
    assert "BLOCKER: stop" in out


def test_banner_risks_are_not_repeated_at_the_bottom():
    """They were appearing twice: once in the banner, once in a trailing Risks list."""
    out = render(risks=["LICENCE BLOCKER (weights): non-commercial clause"])
    assert out.count("non-commercial clause") == 1


def test_agent_freeform_risks_appear_once_under_other_notes():
    out = render(risks=["pinned torch is two years old"])
    assert out.count("pinned torch is two years old") == 1
    assert "## Other notes" in out


def test_no_other_notes_section_when_everything_was_in_the_banner():
    out = render(risks=["HALLUCINATED PATH: ghost.py"])
    assert "## Other notes" not in out


def test_no_empty_risks_section_on_a_clean_repo():
    assert "## Other notes" not in render()


def test_freeform_risks_dropped_when_structured_alerts_exist():
    """Models restate the same finding in both fields; rendering both repeats it."""
    out = render(alerts=[{"severity": "blocker", "title": "Hardcoded ProtT5 path",
                          "detail": "features.py line 39"}],
                 risks=["Hardcoded ProtT5 path /home/x/Prot-T5 in Features/features.py"])
    assert "## Other notes" not in out
    assert out.count("Hardcoded ProtT5 path") == 1


def test_freeform_risks_kept_when_there_are_no_alerts():
    out = render(risks=["pinned torch is two years old"])
    assert "## Other notes" in out and "pinned torch" in out
