"""Serialize GPU trials across workers and release the lock on process exit."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
if os.name == 'nt':
    import msvcrt
else:
    import fcntl

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def gpu_lock(owner, timeout_seconds=1800):
    path = ROOT / 'common/gpu.lock'
    handle = path.open('a+b')
    if path.stat().st_size == 0:
        handle.write(b'0')
        handle.flush()
    started = time.monotonic()
    acquired = False
    try:
        while not acquired:
            handle.seek(0)
            try:
                if os.name == 'nt':
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                if time.monotonic() - started > timeout_seconds:
                    raise TimeoutError('Shared GPU stayed busy past the wait limit.')
                time.sleep(0.5)
        (ROOT / 'common/gpu_owner.json').write_text(json.dumps({
            'owner': str(owner), 'pid': os.getpid(),
            'acquired_utc': datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        yield
    finally:
        if acquired:
            (ROOT / 'common/gpu_owner.json').write_text(json.dumps({
                'owner': None, 'released_utc': datetime.now(timezone.utc).isoformat(),
            }))
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        if acquired:
            # A worker with several short consecutive folds should give another
            # waiting process one polling interval to acquire the shared device.
            time.sleep(0.55)


def research_deadline_utc():
    """Keep this run bounded; a later manual reproduction can set its own deadline."""
    value = os.environ.get('CLOUDAI_RESEARCH_DEADLINE_UTC', '2026-10-10T20:34:15+00:00')
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('CLOUDAI_RESEARCH_DEADLINE_UTC needs an explicit UTC offset.')
    return result.astimezone(timezone.utc)


def deadline_passed():
    return datetime.now(timezone.utc) >= research_deadline_utc()
