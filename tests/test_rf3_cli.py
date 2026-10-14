"""Config composition must work without the source checkout on sys.path."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RF3_SOURCE = ROOT / "models" / "rf3" / "src" / "rf3"
RF3_CONFIGS = ROOT / "models" / "rf3" / "configs"

# Stub only model execution: config discovery and Hydra composition remain real.
COMPOSE_CLI = """
import json
from pathlib import Path
import sys
import types

package_dir = Path(sys.argv[1])
sys.path.insert(0, str(package_dir.parent))
import rf3.cli
from typer.testing import CliRunner

assert Path(rf3.cli.__file__).resolve() == package_dir / "cli.py"
captured = []
inference = types.ModuleType("rf3.inference")
inference.run_inference = captured.append
sys.modules["rf3.inference"] = inference
result = CliRunner().invoke(rf3.cli.app, json.loads(sys.argv[2]))
if result.exception is not None:
    raise result.exception
assert result.exit_code == 0, result.output
assert len(captured) == 1
cfg = captured[0]
assert cfg._target_ == "rf3.inference_engines.rf3.RF3InferenceEngine"
assert cfg.ckpt_path == "run14_confhead"
assert cfg.inputs == "example.json"
assert cfg.verbose is True
print(json.dumps({"num_steps": cfg.num_steps, "inputs": cfg.inputs}))
"""


@pytest.mark.parametrize("layout", ["editable", "wheel"])
@pytest.mark.parametrize(
    "arguments, expected_steps",
    [
        (["fold", "--verbose", "example.json"], 50),
        (["predict", "--verbose", "inputs=example.json", "num_steps=7"], 7),
    ],
)
def test_cli_composes_real_configs_from_any_working_directory(
    tmp_path, layout, arguments, expected_steps
):
    """Exercise both entry points using the actual CLI and shipped YAML files."""
    if layout == "editable":
        package = tmp_path / "checkout" / "models" / "rf3" / "src" / "rf3"
        configs = package.parents[1] / "configs"
    else:
        package = tmp_path / "site-packages" / "rf3"
        configs = package / "configs"
    package.mkdir(parents=True)
    for filename in ("__init__.py", "cli.py", "_config.py"):
        shutil.copy2(RF3_SOURCE / filename, package / filename)
    shutil.copytree(RF3_CONFIGS, configs)
    working_dir = tmp_path / "unrelated-workdir"
    working_dir.mkdir()

    completed = subprocess.run(
        [sys.executable, "-I", "-c", COMPOSE_CLI, str(package), json.dumps(arguments)],
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == {
        "num_steps": expected_steps,
        "inputs": "example.json",
    }


def test_config_lookup_rejects_an_empty_config_directory(tmp_path, monkeypatch):
    from rf3 import _config

    package = tmp_path / "site-packages" / "rf3"
    empty_configs = package / "configs"
    empty_configs.mkdir(parents=True)
    monkeypatch.setattr(_config, "__file__", str(package / "_config.py"))

    with pytest.raises(
        FileNotFoundError, match="RF3 inference.yaml not found"
    ) as error:
        _config.get_rf3_config_dir()

    assert str(empty_configs) in str(error.value)
    assert str(tmp_path / "configs") in str(error.value)
