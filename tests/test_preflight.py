"""Deterministic scan: build detection, version pins, bio stack, and the licence gate."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import preflight  # noqa: E402

MIT = "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy"
AF3 = "AlphaFold 3 Model Parameters Terms of Use\nUse is permitted for non-commercial research purposes only."


def make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return tmp_path


# ------------------------------------------------------------------ licence
def test_permissive_licence_is_redistributable(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {"LICENSE": MIT}))
    assert pf.licence.redistributable is True
    assert not pf.licence.blocking
    assert preflight.licence_warning(pf) == ""


def test_restricted_licence_blocks_and_warns(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {"LICENSE": AF3}))
    assert pf.licence.redistributable is False
    assert pf.licence.blocking
    assert "LICENCE BLOCKER" in preflight.licence_warning(pf)


@pytest.mark.parametrize("text", [
    "This model is released for research use only.",
    "Licensed under CC-BY-NC 4.0",
    "You may not use this for commercial purposes. Not for commercial use.",
    "Llama 3 Community License Agreement",
])
def test_other_restricted_markers(tmp_path, text):
    pf = preflight.scan(make_repo(tmp_path, {"LICENSE": text}))
    assert pf.licence.blocking, f"should block: {text!r}"


def test_missing_licence_is_unknown_not_permissive(tmp_path):
    """Absence of a LICENSE file must not be read as permission to serve."""
    pf = preflight.scan(make_repo(tmp_path, {"README.md": "a model"}))
    assert pf.licence.redistributable is None
    assert not pf.licence.blocking          # unknown, not proven bad
    assert "UNVERIFIED" in preflight.licence_warning(pf)


def test_weights_licence_file_is_found(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {"LICENSE-WEIGHTS": AF3}))
    assert pf.weights_licence.blocking
    assert "LICENCE BLOCKER (weights)" in preflight.licence_warning(pf)


def test_permissive_code_does_not_hide_restricted_weights(tmp_path):
    """The common shape for large bio models, and the one that must not be missed:
    Apache-2.0 source, non-commercial weights. Reading only LICENSE would clear it."""
    pf = preflight.scan(make_repo(tmp_path, {"LICENSE": "MIT License", "MODEL_LICENSE": AF3}))
    assert pf.licence.redistributable is True        # the code is fine
    assert pf.weights_licence.blocking               # the weights are not
    assert pf.effective_licence.blocking
    warning = preflight.licence_warning(pf)
    assert "weights" in warning and "BLOCKER" in warning


def test_external_weights_without_a_licence_file_are_flagged(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {
        "LICENSE": "MIT License",
        "download.py": 'from_pretrained("facebook/esm2_t33_650M_UR50D")\n',
    }))
    assert pf.hf_ids
    assert "check the model card" in pf.weights_licence.reason
    assert "UNVERIFIED (weights)" in preflight.licence_warning(pf)


# ------------------------------------------------------------- build + pins
def test_detects_build_files_and_python_version(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {
        "requirements.txt": "torch==2.4.1\n",
        "Dockerfile": "FROM python:3.11\n",
        "pyproject.toml": '[project]\nrequires-python = ">=3.10"\n',
    }))
    assert {"requirements.txt", "Dockerfile", "pyproject.toml"} <= set(pf.build_files)
    assert pf.python_version == ">=3.10"


def test_extracts_cuda_and_torch_pins(tmp_path):
    """These feed the test-fixer's compatibility matrix."""
    pf = preflight.scan(make_repo(tmp_path, {
        "requirements.txt": "torch==2.4.1\ntorch-geometric==2.5.0\n"
                            "--extra-index-url https://download.pytorch.org/whl/cu121\n",
    }))
    assert pf.cuda_pins["torch"] == "2.4.1"
    assert pf.cuda_pins["cuda"] == "12.1"
    assert "torch" in pf.native_deps


def test_detects_tests(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {"pytest.ini": "", "tests/test_a.py": ""}))
    assert "pytest.ini" in pf.test_markers
    assert "tests" in pf.test_dirs


# ---------------------------------------------------------------- bio + weights
def test_bio_stack_detected(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {"requirements.txt": "biopython\nopenmm\nfair-esm\n"}))
    assert pf.is_bio
    assert "biopython" in pf.bio_deps
    assert "computational-biology" in preflight.to_prompt_block(pf)


def test_non_bio_repo_omits_bio_guidance(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {"requirements.txt": "fastapi\nuvicorn\n"}))
    assert not pf.is_bio
    assert "computational-biology" not in preflight.to_prompt_block(pf)


def test_finds_weight_sources(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {
        "predict.py": 'from transformers import AutoModel\n'
                      'm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")\n'
                      'u = "https://example.com/params/model_1.npz"\n',
    }))
    assert "facebook/esm2_t33_650M_UR50D" in pf.hf_ids
    assert any("model_1.npz" in u for u in pf.weight_urls)
    assert "predict.py" in pf.entrypoints


def test_measures_bundled_weights(tmp_path):
    (tmp_path / "w.safetensors").write_bytes(b"x" * 2_000_000)
    pf = preflight.scan(tmp_path)
    assert pf.bundled_weights_mb == pytest.approx(2.0, abs=0.1)


# -------------------------------------------------------------- robustness
def test_empty_repo_does_not_raise(tmp_path):
    pf = preflight.scan(tmp_path)
    assert pf.build_files == []
    assert preflight.to_prompt_block(pf)          # still renders


def test_malformed_pyproject_does_not_raise(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {"pyproject.toml": "this is [not valid toml"}))
    assert pf.python_version == ""


def test_scan_is_serialisable(tmp_path):
    import json
    pf = preflight.scan(make_repo(tmp_path, {"LICENSE": MIT, "requirements.txt": "torch\n"}))
    assert json.loads(json.dumps(pf.to_dict()))["licence"]["redistributable"] is True


# --------------------------------------------------------- broken scripts
def test_syntax_errors_are_detected(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {
        "good.py": "def f():\n    return 1\n",
        "bad.py": "def f(:\n  pass\n",
    }))
    paths = [b["path"] for b in pf.broken_scripts]
    assert "bad.py" in paths and "good.py" not in paths
    assert "line" in pf.broken_scripts[0]["error"]


def test_python2_only_file_is_reported_as_broken(tmp_path):
    """It will not run on this interpreter, which is the finding that matters."""
    pf = preflight.scan(make_repo(tmp_path, {"old.py": "print 'hello'\n"}))
    assert [b["path"] for b in pf.broken_scripts] == ["old.py"]


def test_a_healthy_repo_reports_no_broken_scripts(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {"a.py": "x = 1\n", "b.py": "y = 2\n"}))
    assert pf.broken_scripts == []


def test_broken_scripts_reach_the_prompt_block(tmp_path):
    pf = preflight.scan(make_repo(tmp_path, {"bad.py": "def (\n"}))
    assert "DO NOT PARSE" in preflight.to_prompt_block(pf)
    assert "bad.py" in preflight.to_prompt_block(pf)
