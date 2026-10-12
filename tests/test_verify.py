"""Agent 1's safety net: a schema-valid analysis is not necessarily a useful one.

Every check here exists because an empty or invented field would otherwise be
carried silently into a stage that provisions a GPU.
"""
from pathlib import Path

import pytest

from pipeline import preflight, verify


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "infer.py").write_text("def predict(x): return x\n")
    (tmp_path / "requirements.txt").write_text("torch==2.1.0+cu121\n")
    (tmp_path / "tests").mkdir()
    return tmp_path


def good_analysis() -> dict:
    return {
        "summary": "a model", "task": "folding",
        "dependency_files": ["requirements.txt"],
        "key_files": [{"path": "src/infer.py", "role": "inference"}],
        "functions": [{"file": "src/infer.py", "name": "predict"}],
        "inference": {"file": "src/infer.py", "function_or_class": "predict",
                      "input_format": "FASTA string", "output_format": "PDB string"},
        "tests": {"framework": "pytest", "command": "pytest -q", "paths": ["tests"]},
        "hardware": {"gpu_required": True, "min_gpu_memory_gb": 24.0, "peak_memory_gb": 24.0},
        "weights": {"source": "huggingface"},
        "license": "MIT",
        "metadata": {"model_name": "demo", "purpose": "fold proteins", "status": "active"},
        "licensing": {"code": "MIT", "weights": "CC-BY-4.0", "commercial_use": "yes"},
        "capabilities": [{"task": "folding", "entrypoint": "src/infer.py",
                          "inputs": "FASTA", "outputs": "PDB"}],
        "scripts": [{"path": "src/infer.py", "kind": "cli", "invocation": "python src/infer.py"}],
        "assets": [{"path": "requirements.txt", "kind": "config", "bundled": True}],
        "inputs": [{"name": "sequence", "format": "fasta"}],
        "outputs": [{"name": "structure", "format": "pdb"}],
        "flow": [{"id": "in", "label": "FASTA", "kind": "input", "next": ["model"]},
                 {"id": "model", "label": "predict", "kind": "model", "next": ["out"]},
                 {"id": "out", "label": "PDB", "kind": "output"}],
        "alerts": [],
    }


def scan(repo: Path):
    return preflight.scan(repo)


# ------------------------------------------------------------- path checking
def test_clean_analysis_produces_no_findings(repo):
    pf = scan(repo)
    pf.cuda_pins, pf.test_markers, pf.test_dirs = {}, [], []
    pf.hf_ids, pf.weight_urls = [], []
    pf.licence.name = "MIT"
    assert verify.review(good_analysis(), pf, repo, hit_step_limit=False) == []


def test_invented_path_is_caught(repo):
    a = good_analysis()
    a["inference"]["file"] = "src/does_not_exist.py"
    missing = verify.missing_paths(a, repo)
    assert ("inference.file", "src/does_not_exist.py") in missing


def test_globs_and_shell_fragments_are_not_treated_as_paths(repo):
    a = good_analysis()
    a["tests"]["paths"] = ["tests/**/*.py", "python -m pytest", "src/infer.py"]
    assert verify.missing_paths(a, repo) == []


def test_leading_dot_slash_and_line_suffix_normalised(repo):
    a = good_analysis()
    a["key_files"] = [{"path": "./src/infer.py", "role": "x"}, {"path": "src/infer.py:12", "role": "y"}]
    assert verify.missing_paths(a, repo) == []


def test_each_missing_path_reported_once(repo):
    a = good_analysis()
    a["key_files"] = [{"path": "ghost.py", "role": "a"}, {"path": "ghost.py", "role": "b"}]
    assert len(verify.missing_paths(a, repo)) == 1


# ------------------------------------------------------------- completeness
@pytest.mark.parametrize("mutate,expected", [
    (lambda a: a["inference"].update(file="", function_or_class=""), "no inference entrypoint"),
    (lambda a: a["inference"].update(input_format=""), "input/output format"),
    (lambda a: a["tests"].update(command=""), "no test command"),
    (lambda a: a["hardware"].update(min_gpu_memory_gb=None), "no VRAM estimate"),
])
def test_downstream_blockers_are_reported(mutate, expected):
    a = good_analysis()
    mutate(a)
    assert any(expected in g for g in verify.completeness_gaps(a))


def test_a_repo_with_no_tests_is_not_a_gap():
    a = good_analysis()
    a["tests"] = {"framework": "none", "command": ""}
    assert not any("test command" in g for g in verify.completeness_gaps(a))


def test_cpu_model_needs_no_vram_estimate():
    a = good_analysis()
    a["hardware"] = {"gpu_required": False, "min_gpu_memory_gb": None}
    assert not any("VRAM" in g for g in verify.completeness_gaps(a))


# ----------------------------------------------------------- contradictions
def test_cuda_pins_versus_no_gpu(repo):
    pf = scan(repo)
    a = good_analysis()
    a["hardware"]["gpu_required"] = False
    assert any("no GPU is required" in c for c in verify.contradictions(a, pf))


def test_scan_found_tests_but_agent_did_not(repo):
    pf = scan(repo)
    pf.test_dirs = ["tests"]
    a = good_analysis()
    a["tests"]["command"] = ""
    assert any("no test command" in c for c in verify.contradictions(a, pf))


def test_weights_referenced_but_not_recorded(repo):
    pf = scan(repo)
    pf.hf_ids = ["facebook/esm2_t33"]
    a = good_analysis()
    a["weights"] = {"source": ""}
    assert any("no weights source" in c for c in verify.contradictions(a, pf))


def test_licence_disagreement_favours_the_file(repo):
    pf = scan(repo)
    pf.licence.name, pf.licence.file = "GPL-3.0", "LICENSE"
    a = good_analysis()
    a["licensing"]["code"] = "MIT"
    assert any("the file wins" in c for c in verify.contradictions(a, pf))


def test_licence_substring_match_is_not_a_disagreement(repo):
    pf = scan(repo)
    pf.licence.name = "Apache-2.0"
    a = good_analysis()
    a["licensing"]["code"] = "Apache-2.0 License"
    assert not any("the file wins" in c for c in verify.contradictions(a, pf))


def test_claiming_commercial_use_against_restricted_weights(repo):
    """The costliest possible mistake: the model waving through terms that forbid serving."""
    pf = scan(repo)
    pf.weights_licence.redistributable = False
    pf.weights_licence.file, pf.weights_licence.reason = "MODEL_LICENSE", "non-commercial clause"
    a = good_analysis()
    a["licensing"]["commercial_use"] = "yes"
    assert any("reads as restricted" in c for c in verify.contradictions(a, pf))


def test_archived_project_called_active(repo):
    pf = scan(repo)
    pf.archived_hint = "no longer maintained"
    a = good_analysis()
    a["metadata"]["status"] = "active"
    assert any("no longer maintained" in c for c in verify.contradictions(a, pf))


def test_bundled_assets_are_checked_but_downloads_are_not(repo):
    a = good_analysis()
    a["assets"] = [{"path": "weights/model.pt", "kind": "weights", "bundled": False},
                   {"path": "configs/missing.yaml", "kind": "config", "bundled": True}]
    missing = dict((p, f) for f, p in verify.missing_paths(a, repo))
    assert "configs/missing.yaml" in missing
    assert "weights/model.pt" not in missing


def test_bare_function_name_entrypoint_is_not_a_path(repo):
    a = good_analysis()
    a["capabilities"] = [{"task": "fold", "entrypoint": "predict"}]
    assert verify.missing_paths(a, repo) == []


# ------------------------------------------------------------------- review
def test_step_limit_is_surfaced_first(repo):
    out = verify.review(good_analysis(), scan(repo), repo, hit_step_limit=True)
    assert out and out[0].startswith("ANALYSIS INCOMPLETE") and "step limit" in out[0]


def test_review_never_raises_on_a_degenerate_analysis(repo):
    """The thinnest thing that still validates against RepoAnalysis."""
    out = verify.review({"summary": "x", "task": "y"}, scan(repo), repo, hit_step_limit=False)
    assert isinstance(out, list) and out


def test_review_survives_wrong_types(repo):
    """An LLM-repaired object can carry nulls where lists were expected."""
    a = {"summary": "x", "task": "y", "key_files": None, "functions": None,
         "inference": {}, "tests": {}, "hardware": {}, "dependency_files": None}
    assert isinstance(verify.review(a, scan(repo), repo, hit_step_limit=False), list)


# ------------------------------------------------- licence name normalisation
@pytest.mark.parametrize("declared,in_file", [
    ("apache-2.0", "Apache License"),                 # the real GraphEC false positive
    ("Apache License 2.0", "Apache-2.0"),
    ("Apache License, Version 2.0", "Apache-2.0"),
    ("MIT", "MIT License"),
    ("BSD-3-Clause", "BSD 3-Clause License"),
])
def test_same_licence_written_differently_is_not_a_contradiction(repo, declared, in_file):
    pf = scan(repo)
    pf.licence.name, pf.licence.file = in_file, "LICENSE"
    a = good_analysis()
    a["licensing"]["code"] = declared
    assert not any("the file wins" in c for c in verify.contradictions(a, pf))


@pytest.mark.parametrize("declared,in_file", [
    ("MIT", "Apache License"),
    ("GPL-3.0", "MIT License"),
    ("BSD-3-Clause", "Mozilla Public License"),
])
def test_genuinely_different_licences_still_flagged(repo, declared, in_file):
    pf = scan(repo)
    pf.licence.name, pf.licence.file = in_file, "LICENSE"
    a = good_analysis()
    a["licensing"]["code"] = declared
    assert any("the file wins" in c for c in verify.contradictions(a, pf))


# --- the EC2 planner's own prerequisites ------------------------------------
# Each of these was a real gap found by comparing two analyses of the same repo:
# the thin one omitted exactly these and still validated.

def test_execution_flag_listed_as_input_is_flagged():
    a = good_analysis()
    a["inputs"].append({"name": "gpu", "format": "int", "required": True})
    out = verify._io_problems(a)
    assert any("execution flag" in m for m in out)
    assert any("serving schema" in m for m in out)


def test_real_inputs_are_not_flagged():
    assert verify._io_problems(good_analysis()) == []


@pytest.mark.parametrize("names", [["text", "text", "text"], ["json", "csv"]])
def test_outputs_named_by_format_are_flagged(names):
    a = good_analysis()
    a["outputs"] = [{"name": n, "format": n} for n in names]
    assert any("named by format" in m for m in verify._io_problems(a))


def test_outputs_named_by_artefact_pass():
    a = good_analysis()
    a["outputs"] = [{"name": "example_top5.txt"}, {"name": "scores.json"}]
    assert verify._io_problems(a) == []


def test_missing_peak_memory_is_a_gap_when_gpu_required():
    a = good_analysis()
    a["hardware"]["peak_memory_gb"] = None
    assert any("peak memory" in g for g in verify.completeness_gaps(a))


def test_peak_memory_not_required_without_gpu():
    a = good_analysis()
    a["hardware"] = {"gpu_required": False}
    assert not any("peak memory" in g for g in verify.completeness_gaps(a))


def test_missing_call_path_is_a_gap():
    a = good_analysis()
    a["functions"] = []
    assert any("call path" in g for g in verify.completeness_gaps(a))


def test_readme_pins_the_analysis_ignored_are_reported(repo: Path):
    (repo / "README.md").write_text(
        "# demo\n\nDeveloped with Python 3.8.16, pytorch v1.13.1 and scipy 1.10.1.\n")
    pf = preflight.scan(repo)
    assert pf.readme_pins, "preflight should read pins out of README prose"
    a = good_analysis()
    a["version_pins"] = []
    assert any("version pin" in c for c in verify.contradictions(a, pf))


def test_recorded_pins_satisfy_the_check(repo: Path):
    (repo / "README.md").write_text("# demo\n\nDeveloped with Python 3.8.16.\n")
    pf = preflight.scan(repo)
    a = good_analysis()
    a["version_pins"] = [{"package": "Python", "version": "3.8.16", "confirmed": False}]
    assert not any("version pin" in c for c in verify.contradictions(a, pf))


def test_bundled_linux_binary_without_platform_constraint_is_reported(repo: Path):
    binary = repo / "tools" / "mkdssp"
    binary.parent.mkdir()
    binary.write_bytes(b"\x7fELF\x02\x01\x01" + b"\x00" * 64)
    binary.chmod(0o755)
    pf = preflight.scan(repo)
    assert pf.bundled_binaries and pf.bundled_binaries[0]["format"] == "ELF (Linux)"
    assert any("platform constraint" in c for c in verify.contradictions(good_analysis(), pf))


def test_declared_platform_constraint_clears_it(repo: Path):
    binary = repo / "tools" / "mkdssp"
    binary.parent.mkdir()
    binary.write_bytes(b"\x7fELF\x02\x01\x01" + b"\x00" * 64)
    binary.chmod(0o755)
    pf = preflight.scan(repo)
    a = good_analysis()
    a["hardware"]["platform_constraint"] = "Linux x86-64: bundled ELF mkdssp"
    assert not any("platform constraint" in c for c in verify.contradictions(a, pf))


def test_python_source_is_not_mistaken_for_a_binary(repo: Path):
    script = repo / "run.sh"
    script.write_text("#!/bin/bash\necho hi\n")
    script.chmod(0o755)
    pf = preflight.scan(repo)
    assert pf.bundled_binaries == []


# --- shorthand vs hallucination ---------------------------------------------
# "fold0-4.ckpt" names five real files. Reporting it as a hallucination put
# "treat the analysis as unreliable" on top of a sound report, which trains the
# reader to ignore the banner that matters.

@pytest.fixture
def ckpt_repo(repo: Path) -> Path:
    """The standard repo, plus a five-fold checkpoint directory."""
    model = repo / "model"
    model.mkdir()
    for i in range(5):
        (model / f"fold{i}.ckpt").write_text("w")
    return repo


@pytest.mark.parametrize("cited", ["model/fold0-4.ckpt", "model/fold{0..4}.ckpt"])
def test_range_shorthand_is_not_a_hallucination(ckpt_repo: Path, cited: str):
    a = good_analysis()
    a["assets"] = [{"path": cited, "kind": "weights", "bundled": True}]
    assert missing_names(verify.missing_paths(a, ckpt_repo)) == []
    assert missing_names(verify.shorthand_paths(a, ckpt_repo)) == [cited]


def test_shorthand_is_reported_but_not_as_unreliable(ckpt_repo: Path):
    a = good_analysis()
    a["assets"] = [{"path": "model/fold0-4.ckpt", "kind": "weights", "bundled": True}]
    risks = verify.review(a, preflight.scan(ckpt_repo), ckpt_repo, hit_step_limit=False)
    assert any(r.startswith("IMPRECISE PATH") for r in risks)
    assert not any("HALLUCINATED" in r for r in risks)


def test_shorthand_in_a_directory_that_does_not_exist_is_still_a_hallucination(ckpt_repo: Path):
    a = good_analysis()
    a["assets"] = [{"path": "nope/fold0-4.ckpt", "kind": "weights", "bundled": True}]
    assert missing_names(verify.missing_paths(a, ckpt_repo)) == ["nope/fold0-4.ckpt"]
    assert verify.shorthand_paths(a, ckpt_repo) == []


def test_a_genuinely_invented_file_is_still_caught(ckpt_repo: Path):
    a = good_analysis()
    a["assets"] = [{"path": "model/imaginary.ckpt", "kind": "weights", "bundled": True}]
    assert missing_names(verify.missing_paths(a, ckpt_repo)) == ["model/imaginary.ckpt"]


def missing_names(pairs):
    return [p for _field, p in pairs]


def test_angle_bracket_placeholders_are_ignored_entirely(ckpt_repo: Path):
    """`<ID>.ckpt` is filtered before the existence check, so it is neither a
    hallucination nor a shorthand finding - it is simply not a path claim."""
    a = good_analysis()
    a["assets"] = [{"path": "model/<ID>.ckpt", "kind": "weights", "bundled": True}]
    assert verify.missing_paths(a, ckpt_repo) == []
    assert verify.shorthand_paths(a, ckpt_repo) == []
