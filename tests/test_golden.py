"""Golden-output checks: assert the model's numbers, not just its status code."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "serving"))
import contract_tests  # noqa: E402

BODY = {
    "request_id": "r-1",
    "outputs": [{"label": "enzyme", "score": 0.91, "mean_plddt": 86.4}],
    "model": {"name": "esmfold", "version": "1"},
}


# ------------------------------------------------------------------- _select
@pytest.mark.parametrize("path,expected", [
    ("outputs.0.score", 0.91),
    ("outputs.0.label", "enzyme"),
    ("model.name", "esmfold"),
    ("request_id", "r-1"),
])
def test_select_walks_dicts_and_lists(path, expected):
    assert contract_tests._select(BODY, path) == (True, expected)


@pytest.mark.parametrize("path", ["nope", "outputs.5.score", "outputs.x", "model.name.deeper"])
def test_select_reports_missing(path):
    found, _ = contract_tests._select(BODY, path)
    assert found is False


# -------------------------------------------------------------- _check_golden
def test_range_passes_inside_bounds():
    assert contract_tests._check_golden(BODY, {"path": "outputs.0.mean_plddt", "min": 70, "max": 100}) == ""


def test_range_fails_below_min():
    msg = contract_tests._check_golden(BODY, {"path": "outputs.0.mean_plddt", "min": 90})
    assert "< min 90" in msg


def test_range_fails_above_max():
    assert "> max" in contract_tests._check_golden(BODY, {"path": "outputs.0.score", "max": 0.5})


def test_missing_path_fails():
    assert "not present" in contract_tests._check_golden(BODY, {"path": "outputs.0.plddt", "min": 1})


def test_equals_len_contains():
    assert contract_tests._check_golden(BODY, {"path": "outputs.0.label", "equals": "enzyme"}) == ""
    assert "expected" in contract_tests._check_golden(BODY, {"path": "outputs.0.label", "equals": "other"})
    assert contract_tests._check_golden(BODY, {"path": "outputs", "len": 1}) == ""
    assert "length" in contract_tests._check_golden(BODY, {"path": "outputs", "len": 2})
    assert contract_tests._check_golden(BODY, {"path": "outputs.0.label", "contains": "enz"}) == ""
    assert "not in" in contract_tests._check_golden(BODY, {"path": "outputs.0.label", "contains": "zzz"})


def test_numeric_check_on_non_number_fails_clearly():
    """A string where a number is expected is a real regression, not a pass."""
    msg = contract_tests._check_golden(BODY, {"path": "outputs.0.label", "min": 1})
    assert "expected a number" in msg


def test_bool_is_not_accepted_as_number():
    msg = contract_tests._check_golden({"flag": True}, {"path": "flag", "min": 0, "max": 2})
    assert "expected a number" in msg


def test_empty_path_checks_whole_body():
    assert contract_tests._check_golden(BODY, {"path": "", "contains": "esmfold"}) == ""


# ------------------------------------------------------- run_contract_tests
def _send_ok(_payload):
    return 200, BODY, 5.0


def test_golden_case_recorded_as_passing():
    cases = []
    contract_tests._run_golden(_send_ok, lambda n, k, ok, d="", ms=None: cases.append((n, k, ok, d)), 0,
                               {"name": "ref", "request": {"inputs": ["M"]},
                                "checks": [{"path": "outputs.0.score", "min": 0.5, "max": 1.0}]})
    assert cases[0][2] is True and cases[0][1] == "golden"


def test_golden_case_fails_on_drifted_number():
    cases = []
    contract_tests._run_golden(_send_ok, lambda n, k, ok, d="", ms=None: cases.append((n, k, ok, d)), 0,
                               {"name": "ref", "request": {"inputs": ["M"]},
                                "checks": [{"path": "outputs.0.score", "min": 0.95}]})
    assert cases[0][2] is False
    assert "< min" in cases[0][3]


def test_golden_case_fails_on_non_200():
    cases = []
    contract_tests._run_golden(lambda p: (500, {"message": "boom"}, 1.0),
                               lambda n, k, ok, d="", ms=None: cases.append((n, k, ok, d)), 0,
                               {"request": {"inputs": ["M"]}, "checks": []})
    assert cases[0][2] is False and "status 500" in cases[0][3]


def test_malformed_golden_case_is_a_failure_not_a_crash():
    cases = []
    contract_tests._run_golden(_send_ok, lambda n, k, ok, d="", ms=None: cases.append((n, k, ok, d)), 0,
                               {"name": "no request"})
    assert cases[0][2] is False


def test_samples_without_golden_key_still_work():
    """Backward compatibility: existing sample_requests.json files have no 'golden'."""
    schema = json.loads((ROOT / "schema" / "endpoint_schema.json").read_text())
    out = contract_tests.run_contract_tests(_send_ok, None, schema, {"valid": [], "invalid": []})
    assert out["total"] == 0 and out["failed"] == 0
