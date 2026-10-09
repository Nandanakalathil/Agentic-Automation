"""The endpoint contract: strict for typed biological payloads, permissive otherwise.

The permissive half matters as much as the strict half - a non-biological model
(text classifier, image model) must keep working against the same schema.
"""
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "schema" / "endpoint_schema.json").read_text())


def v(name: str) -> Draft202012Validator:
    return Draft202012Validator({"$ref": f"#/$defs/{name}", "$defs": SCHEMA["$defs"]})


REQ, RESP = v("request"), v("response")
PROTEIN = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


def req(inputs, **kw):
    return {"inputs": inputs, **kw}


# ------------------------------------------------- backward compatibility
@pytest.mark.parametrize("payload", [
    {"request_id": "r-1", "inputs": ["hello world"]},
    {"inputs": ["a", "bb", "ccc"], "parameters": {"top_k": 3}},
    {"request_id": "r-bad", "inputs": [123]},           # schema-valid; the adapter rejects it
    {"inputs": [{"text": "no type key", "n": 2}]},      # free-form object
    {"inputs": [[1, 2, 3]]},                            # numeric array
    {"inputs": [True, None, 1.5]},
])
def test_non_biological_inputs_still_accepted(payload):
    """Existing non-bio models must not break on the new schema."""
    assert REQ.is_valid(payload), list(REQ.iter_errors(payload))[:1]


def test_schema_still_exposes_the_three_keys_the_pipeline_reads():
    assert {"request", "response", "error"} <= set(SCHEMA["$defs"])


# --------------------------------------------------------- sequence input
def test_protein_sequence_accepted():
    assert REQ.is_valid(req([{"type": "protein", "sequence": PROTEIN, "id": "egfr-b1"}]))


@pytest.mark.parametrize("bad", [
    {"type": "protein", "sequence": "MKT123VLA"},        # digits are not residues
    {"type": "protein", "sequence": "MKT VLA"},          # whitespace must be stripped
    {"type": "protein", "sequence": ""},                 # empty
    {"type": "protein"},                                 # no sequence
    {"type": "protein", "sequence": PROTEIN, "bogus": 1},  # unknown field
    {"type": "peptide", "sequence": PROTEIN},            # unknown type
])
def test_malformed_sequences_rejected(bad):
    assert not REQ.is_valid(req([bad]))


def test_gaps_and_stop_codons_allowed():
    assert REQ.is_valid(req([{"type": "protein", "sequence": "MKT-VLA*"}]))
    assert REQ.is_valid(req([{"type": "dna", "sequence": "ACGTNNRY"}]))


# -------------------------------------------------------- structure input
def test_structure_inline_or_by_uri():
    assert REQ.is_valid(req([{"type": "structure", "format": "pdb", "data": "ATOM  1  N  ..."}]))
    assert REQ.is_valid(req([{"type": "structure", "format": "mmcif", "uri": "s3://bucket/x.cif"}]))


def test_structure_needs_exactly_one_source():
    """Neither is unusable; both is ambiguous about which wins."""
    assert not REQ.is_valid(req([{"type": "structure", "format": "pdb"}]))
    assert not REQ.is_valid(req([{"type": "structure", "format": "pdb",
                                  "data": "ATOM", "uri": "s3://b/x.pdb"}]))


def test_structure_rejects_bad_format_and_scheme():
    assert not REQ.is_valid(req([{"type": "structure", "format": "xyz", "data": "x"}]))
    assert not REQ.is_valid(req([{"type": "structure", "format": "pdb", "uri": "ftp://b/x.pdb"}]))


def test_structure_chains():
    assert REQ.is_valid(req([{"type": "structure", "format": "pdb", "data": "ATOM", "chains": ["A", "B"]}]))


# -------------------------------------------------------- msa / molecule
def test_msa_and_molecule():
    assert REQ.is_valid(req([{"type": "msa", "format": "a3m", "uri": "s3://b/t.a3m", "depth": 512}]))
    assert REQ.is_valid(req([{"type": "molecule", "format": "smiles", "data": "CC(=O)Oc1ccccc1C(=O)O"}]))
    assert not REQ.is_valid(req([{"type": "msa", "format": "a3m"}]))          # no data or uri
    assert not REQ.is_valid(req([{"type": "molecule", "format": "smiles"}]))  # no data


def test_mixed_batch_of_types():
    """A binder-design request: target structure + designed sequence + its MSA."""
    assert REQ.is_valid(req([
        {"type": "structure", "format": "pdb", "data": "ATOM ...", "chains": ["A"]},
        {"type": "protein", "sequence": PROTEIN},
        {"type": "msa", "format": "a3m", "uri": "s3://b/target.a3m"},
    ]))


# ----------------------------------------------------------------- outputs
def _resp(outputs):
    return {"request_id": "r", "model": {"name": "m", "version": "1"},
            "outputs": outputs, "latency_ms": 12.0}


def test_structure_output_with_confidence():
    assert RESP.is_valid(_resp([{"type": "structure", "format": "pdb", "data": "ATOM",
                                 "plddt": [80.1, 91.2], "mean_plddt": 85.6, "ptm": 0.8}]))


@pytest.mark.parametrize("bad", [
    {"type": "structure", "format": "pdb", "data": "A", "plddt": [120]},   # pLDDT is 0-100
    {"type": "structure", "format": "pdb", "data": "A", "ptm": 1.4},       # pTM is 0-1
    {"type": "structure", "format": "pdb", "data": "A", "mean_plddt": -1},
])
def test_confidence_ranges_enforced(bad):
    assert not RESP.is_valid(_resp([bad]))


def test_sequence_and_scores_outputs():
    assert RESP.is_valid(_resp([{"type": "protein", "sequence": PROTEIN, "score": -1.2,
                                 "per_residue": [0.1, 0.2], "metrics": {"perplexity": 3.1}}]))
    assert RESP.is_valid(_resp([{"type": "scores", "label": "1.1.1.1", "score": 0.93,
                                 "labels": [{"label": "1.1.1.1", "score": 0.93}]}]))


def test_untyped_outputs_still_allowed():
    assert RESP.is_valid(_resp(["a string", 1, {"free": "form"}]))


def test_response_still_requires_core_fields():
    assert not RESP.is_valid({"request_id": "r", "outputs": []})     # no model / latency_ms


def test_warnings_field_optional():
    r = _resp([1]); r["warnings"] = ["truncated MSA to depth 512"]
    assert RESP.is_valid(r)
