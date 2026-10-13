"""Bundle a saved, executed notebook and its actual cloud run."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--notebook',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    notebook=args.notebook.resolve()
    run=args.run.resolve()
    if not notebook.is_relative_to(ROOT) or not run.is_relative_to(ROOT):
        raise ValueError('Choose the saved notebook and run folder inside this project.')
    record=json.loads((run/'run_record.json').read_text())
    if record['execution']!='AWS notebook instance' or record['status']!='complete':
        raise ValueError('This is not a completed AWS run. Keep local checks separate.')
    content=json.loads(notebook.read_text())
    executed=[c for c in content['cells'] if c['cell_type']=='code' and c.get('execution_count') is not None]
    if len(executed)<2 or any(o.get('output_type')=='error' for c in executed for o in c.get('outputs',[])):
        raise ValueError('Save the successfully executed notebook before exporting.')
    with ZipFile(args.output,'w',ZIP_DEFLATED) as archive:
        archive.write(notebook,notebook.relative_to(ROOT).as_posix())
        for path in run.rglob('*'):
            if path.is_file():
                archive.write(path,path.relative_to(ROOT).as_posix())
    print(args.output)

if __name__=='__main__':
    main()
