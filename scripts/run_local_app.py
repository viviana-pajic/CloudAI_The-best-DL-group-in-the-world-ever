"""Start both model APIs in their own environments and serve the shared app."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from gateway import serve


def environment_python(folder):
    return ROOT / folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8780)
    args = parser.parse_args()
    mushroom_port, bike_port = args.port + 1, args.port + 2
    mushroom_python = Path(os.environ.get('MUSHROOM_PYTHON', environment_python('.venv-pycaret')))
    bike_python = Path(os.environ.get('CITIBIKE_PYTHON', environment_python('.venv')))
    for path in [mushroom_python, bike_python, ROOT / 'models/mushroom_candidate.joblib', ROOT / 'models/citibike_candidate.joblib']:
        if not path.exists():
            raise SystemExit(f'Missing {path}. Follow the setup and bundle-rebuild instructions first.')
    for port in [args.port, mushroom_port, bike_port]:
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise SystemExit(f'Port {port} is already occupied; choose another --port value.')
    logs = ROOT / '.tools/app-logs'
    logs.mkdir(parents=True, exist_ok=True)
    children, streams = [], []
    try:
        for name, python, script, port, route in [
            ('mushrooms', mushroom_python, 'app/server.py', mushroom_port, '/model-info'),
            ('citibike', bike_python, 'app/citibike_server.py', bike_port, '/bike/model-info'),
        ]:
            stream = (logs / f'{name}.log').open('w', encoding='utf-8')
            streams.append(stream)
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            child = subprocess.Popen([str(python), str(ROOT / script), '--port', str(port)],
                                     cwd=ROOT, stdout=stream, stderr=stream, creationflags=flags)
            children.append(child)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f'{name} API stopped. See {logs / (name + ".log")}.')
                try:
                    with urlopen(f'http://127.0.0.1:{port}{route}', timeout=1):
                        break
                except (URLError, TimeoutError):
                    time.sleep(.2)
            else:
                raise RuntimeError(f'{name} API did not become ready.')
        serve(args.host, args.port, mushroom_port, bike_port)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for stream in streams:
            stream.close()
