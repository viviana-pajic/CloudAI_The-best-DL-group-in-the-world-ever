"""Execute Citi Bike notebooks 03-10; optionally reproduce the searches."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tools/matplotlib-citibike'))
os.environ.setdefault('IPYTHONDIR', str(ROOT / '.tools/ipython-citibike'))

import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output

NAMES = ['03_data_preparation.ipynb', '04_baselines.ipynb', '05_automl.ipynb',
         '06_ridge.ipynb', '07_xgboost.ipynb', '08_random_forest.ipynb',
         '09_error_analysis.ipynb', '10_model_comparison.ipynb']


if __name__ == '__main__':
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Use the documented Python 3.12 Citi Bike environment.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='Rerun searches; XGBoost requires CUDA.')
    parser.add_argument('--only', nargs='+', choices=NAMES)
    args = parser.parse_args()
    os.environ['CITIBIKE_RETRAIN'] = '1' if args.train else '0'
    os.chdir(ROOT)
    shell = TerminalInteractiveShell.instance()
    shell.display_formatter.active_types = ['text/plain', 'text/html', 'image/png']
    for name in args.only or NAMES:
        notebook = nbformat.read(ROOT / 'citibike' / name, as_version=4)
        print(f'Executing {name}; mode={"training" if args.train else "review saved experiments"}', flush=True)
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                execution = shell.run_cell(cell.source, store_history=True)
            if not execution.success:
                raise RuntimeError(f'{name}: {execution.error_before_exec or execution.error_in_exec}\n{captured.stdout}\n{captured.stderr}')
            cell.execution_count = execution.execution_count
            cell.outputs = []
            for kind, value in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
                if value:
                    cell.outputs.append(nbformat.v4.new_output('stream', name=kind, text=value))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
        notebook.metadata['execution_environment'] = {
            'python': sys.version.split()[0], 'mode': 'training' if args.train else 'review saved experiments',
            'runner': 'Notebook code cells executed in order through IPython, with captured outputs.',
        }
        nbformat.validate(notebook)
        nbformat.write(notebook, ROOT / 'citibike' / name)
    print('Saved executed notebooks. The AWS notebook is excluded until it runs on AWS.', flush=True)
