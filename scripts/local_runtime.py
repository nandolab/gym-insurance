"""Launch a module using the desktop runtime without changing shared installations.

    python -s scripts/local_runtime.py -m pytest
    python -s scripts/local_runtime.py -m gym_insurance.benchmark --help
"""
import runpy
import os
import site
import sys
import uuid
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
scoped = root / ".local" / "runtime-deps"
if scoped.is_dir():
    sys.path.insert(0, str(scoped))
# Appended after bundled site-packages: existing torch/gymnasium/pytest are read,
# while the bundled numpy/pandas and the scoped missing dependency stay preferred.
user_site = site.getusersitepackages()
if isinstance(user_site, str) and Path(user_site).is_dir() and user_site not in sys.path:
    sys.path.append(user_site)
if len(sys.argv) < 3 or sys.argv[1] != "-m":
    raise SystemExit("usage: python -s scripts/local_runtime.py -m MODULE [arguments]")
module = sys.argv[2]
if module == "pytest":
    # Unrelated user-installed pytest plugins can trigger imports/network clients.
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    if not any(arg.startswith("--basetemp") for arg in sys.argv[3:]):
        temp_parent = root / "artifacts" / "pytest-temp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        sys.argv.append("--basetemp=" + str(temp_parent / uuid.uuid4().hex))
if module == "pip":
    # The desktop sandbox's default temp ACL can reject pip's build tracker.
    pip_temp = root / "artifacts" / "pip-temp" / uuid.uuid4().hex
    pip_temp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(pip_temp)
    os.environ["TEMP"] = os.environ["TMP"] = str(pip_temp)
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
sys.argv = [module, *sys.argv[3:]]
runpy.run_module(module, run_name="__main__", alter_sys=True)
