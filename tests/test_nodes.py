"""Node helpers that generate remote paths and bash (no AWS)."""
import subprocess

from pipeline.config import load_config
from pipeline.nodes import Nodes

STATE = {"sagemaker_plan": {"weights_strategy": "model_tar_gz", "model_artifact_dir": "weights/v1",
                            "max_concurrency": 2, "environment": {"HF_HOME": "/tmp/hf"}}}


def test_relative_artifact_dir_resolves_against_repo():
    assert Nodes._artifact_dir(STATE["sagemaker_plan"]) == "/opt/work/repo/weights/v1"
    assert Nodes._artifact_dir({"weights_strategy": "model_tar_gz", "model_artifact_dir": "/opt/work/model"}) == "/opt/work/model"
    assert Nodes._artifact_dir({"weights_strategy": "baked_into_image", "model_artifact_dir": "w"}) is None


def test_container_script_is_valid_bash_and_uses_python3():
    script = Nodes(load_config())._container_test_script(STATE)
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    assert "-v /opt/work/repo/weights/v1:/opt/ml/model:ro" in script
    assert "MODEL_ARTIFACT_DIR_MISSING" in script
    assert "python3 .mdp/contract_tests.py" in script and "\npython " not in script
    assert "{{.State.Running}}" in script
