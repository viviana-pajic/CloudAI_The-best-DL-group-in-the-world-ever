"""Execute the four current mushroom notebooks in order using this interpreter.

Usage: .venv-pycaret/Scripts/python.exe scripts/run_mushroom_notebooks.py
Historical/EDA snapshots are deliberately not rerun by this entry point.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".tools/jupyter-runtime"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".tools/ipython"))

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client import KernelManager


class CurrentPythonKernel(KernelSpecManager):
    """Avoid accidentally using a global Python kernel with different libraries."""
    def get_kernel_spec(self, kernel_name):
        spec = super().get_kernel_spec(kernel_name)
        spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        return spec


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Use the documented Python 3.11 mushroom environment.")
    names = ["01_data_preparation.ipynb", "02_baseline_models.ipynb",
             "03_automl.ipynb", "04_model_comparison.ipynb"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=names, help="Execute only selected current notebooks.")
    args = parser.parse_args()
    for name in args.only or names:
        path = ROOT / "mushrooms" / name
        notebook = nbformat.read(path, as_version=4)
        print(f"Executing {name} with {sys.executable}", flush=True)
        manager = KernelManager(kernel_name="python3", kernel_spec_manager=CurrentPythonKernel())
        client = NotebookClient(notebook, timeout=600, kernel_name="python3", km=manager,
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute(cleanup_kc=True)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f"Saved executed outputs: {name}", flush=True)
