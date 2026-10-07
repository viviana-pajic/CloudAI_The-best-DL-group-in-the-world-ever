"""Reproducible acquisition and auditing of selected NYC trip histories."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'citibike/source_config.json'
RAW_DIR = ROOT / 'data/raw/citibike'
PROCESSED_DIR = ROOT / 'data/processed'
REPORT_DIR = ROOT / 'reports/citibike'
AUDIT_VERSION = 3


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_archive(url, expected_sha256=None):
    """Cache an official ZIP; interrupted downloads remain .part files."""
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.netloc != 's3.amazonaws.com'
            or parsed.query or parsed.fragment
            or not re.fullmatch(r'/tripdata/[A-Za-z0-9_.-]+\.zip', parsed.path)):
        raise ValueError('Expected an official tripdata bucket HTTPS ZIP URL.')
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / parsed.path.rsplit('/', 1)[-1]
    metadata_path = path.with_suffix('.download.json')
    if not path.exists():
        partial = path.with_suffix('.zip.part')
        digest = hashlib.sha256()
        written = last_report = 0
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as out:
            size = int(response.headers.get('Content-Length', 0))
            metadata = {
                'url': url, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                'last_modified': response.headers.get('Last-Modified'),
                'etag': response.headers.get('ETag'), 'http_content_length': size,
            }
            while block := response.read(8 * 1024 * 1024):
                out.write(block)
                digest.update(block)
                written += len(block)
                if written - last_report >= 128 * 1024 * 1024:
                    print(f'Downloaded {written / 1024**2:,.0f} MiB of {size / 1024**2:,.0f}', flush=True)
                    last_report = written
        if size and written != size:
            raise ValueError('Incomplete download: HTTP length differs from received bytes.')
        metadata.update(bytes=written, sha256=digest.hexdigest())
        if expected_sha256 and metadata['sha256'] != expected_sha256:
            raise ValueError('Downloaded source differs from the pinned SHA-256.')
        if not zipfile.is_zipfile(partial):
            raise ValueError('Downloaded file is not a ZIP archive.')
        partial.replace(path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    else:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {'url': url}
        metadata.update(bytes=path.stat().st_size, sha256=sha256_file(path))
        if expected_sha256 and metadata['sha256'] != expected_sha256:
            raise ValueError('Cached source differs from the pinned SHA-256.')
        print(f'Using verified cached archive: {path.name}', flush=True)
    return path, metadata


def _csv_members(archive, prefix=''):
    """Stream all CSVs, including nested ZIPs, without extracting archive paths."""
    for info in sorted(archive.infolist(), key=lambda item: item.filename):
        name = info.filename
        if info.is_dir() or '__MACOSX' in Path(name).parts or Path(name).name.startswith('._'):
            continue
        virtual_name = prefix + name
        if name.lower().endswith('.csv'):
            with archive.open(info) as stream:
                yield virtual_name, info, stream
        elif name.lower().endswith('.zip'):
            with tempfile.TemporaryFile() as nested, archive.open(info) as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    nested.write(block)
                nested.seek(0)
                with zipfile.ZipFile(nested) as child:
                    yield from _csv_members(child, virtual_name + '!/')


def _json_counts(counter):
    return {str(key): int(value) for key, value in sorted(counter.items())}


@contextmanager
def _open_archives(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    def iterate():
        for path in paths:
            with zipfile.ZipFile(path) as archive:
                yield from _csv_members(archive, Path(path).name + '!/')
    members = iterate()
    try:
        yield members
    finally:
        members.close()


def audit_archive(path, year, chunksize=200_000):
    """Audit every row and member; count starts by their printed NYC dates."""
    day_counts, hour_counts, missing, issues = (Counter() for _ in range(4))
    categories = {name: Counter() for name in ('rideable_type', 'member_casual')}
    schemas, files, id_parts, outside_samples = {}, [], [], []
    minimum_start = maximum_start = None
    total = 0
    expected_start = pd.Timestamp(year=year, month=1, day=1)
    expected_end = pd.Timestamp(year=year + 1, month=1, day=1)
    with _open_archives(path) as members:
        for member, info, stream in members:
            print(f'Auditing {member}', flush=True)
            file_rows, file_dates, file_outside = 0, set(), 0
            file_min = file_max = None
            for frame in pd.read_csv(stream, chunksize=chunksize, dtype='string'):
                columns = tuple(frame.columns)
                if not {'ride_id', 'started_at', 'ended_at'}.issubset(columns):
                    raise ValueError(f'{member}: unsupported schema {columns}; map explicitly first.')
                schemas.setdefault(columns, set()).add(member)
                missing.update({col: int(frame[col].isna().sum()) for col in columns})
                for col, values in categories.items():
                    if col in frame:
                        values.update(frame[col].fillna('<missing>').value_counts().to_dict())
                started = pd.to_datetime(frame['started_at'], format='mixed', errors='coerce')
                ended = pd.to_datetime(frame['ended_at'], format='mixed', errors='coerce')
                if started.dt.tz is not None or ended.dt.tz is not None:
                    raise ValueError('Timezone-aware timestamps need a revised explicit policy.')
                valid = started.notna() & started.ge(expected_start) & started.lt(expected_end)
                issues['unparsed_start'] += int(started.isna().sum())
                issues['unparsed_end'] += int(ended.isna().sum())
                issues['outside_selected_year'] += int((started.notna() & ~valid).sum())
                file_outside += int((started.notna() & ~valid).sum())
                if len(outside_samples) < 10:
                    sample = frame.loc[started.notna() & ~valid, ['ride_id', 'started_at', 'ended_at']].head(10 - len(outside_samples))
                    outside_samples.extend(sample.to_dict(orient='records'))
                month_match = re.search(r'(20\d{2})(\d{2})-citibike', Path(member).name)
                if month_match:
                    label_year, label_month = map(int, month_match.groups())
                    issues['start_outside_filename_month'] += int((started.notna() &
                        ~(started.dt.year.eq(label_year) & started.dt.month.eq(label_month))).sum())
                    issues['end_outside_filename_month'] += int((ended.notna() &
                        ~(ended.dt.year.eq(label_year) & ended.dt.month.eq(label_month))).sum())
                duration = (ended - started).dt.total_seconds()
                issues['negative_clock_duration'] += int(duration.lt(0).sum())
                issues['zero_clock_duration'] += int(duration.eq(0).sum())
                issues['under_60_second_clock_duration'] += int((duration.ge(0) & duration.lt(60)).sum())
                issues['over_24_hour_clock_duration'] += int(duration.gt(86400).sum())
                dates, hours = started.dt.normalize(), started.dt.hour
                if year == 2023:
                    issues['spring_dst_nonexistent_clock_hour'] += int((dates.eq(pd.Timestamp('2023-03-12')) & hours.eq(2)).sum())
                    issues['autumn_dst_ambiguous_clock_hour'] += int((dates.eq(pd.Timestamp('2023-11-05')) & hours.eq(1)).sum())
                ride_ids = frame['ride_id']
                valid_ids = ride_ids.str.fullmatch(r'[0-9A-Fa-f]{16}', na=False)
                issues['missing_or_non_hex_ride_id'] += int((~valid_ids).sum())
                if not valid_ids.all():
                    raise ValueError(f'{member}: ride IDs need a revised duplicate-audit policy.')
                id_parts.append(np.fromiter((int(value, 16) for value in ride_ids), dtype=np.uint64, count=len(frame)))
                counts = dates[valid].value_counts().to_dict()
                day_counts.update(counts)
                file_dates.update(counts)
                hour_counts.update(hours[valid].value_counts().to_dict())
                low, high = started.min(), started.max()
                if pd.notna(low):
                    file_min = low if file_min is None else min(file_min, low)
                    file_max = high if file_max is None else max(file_max, high)
                file_rows += len(frame)
            if not file_rows:
                raise ValueError(f'Empty CSV member: {member}')
            files.append({
                'member': member, 'rows': file_rows, 'uncompressed_bytes': info.file_size,
                'crc32': f'{info.CRC:08x}', 'first_started_at': str(file_min),
                'last_started_at': str(file_max), 'distinct_start_dates': len(file_dates),
                'outside_selected_year': file_outside,
                'counted_rides': file_rows - file_outside,
            })
            total += file_rows
            minimum_start = file_min if minimum_start is None else min(minimum_start, file_min)
            maximum_start = file_max if maximum_start is None else max(maximum_start, file_max)
            print(f'  {file_rows:,} rows; running total {total:,}', flush=True)
    if not files:
        raise ValueError('No CSV members found, including nested archives.')
    print('Checking ride IDs across all files and chunks...', flush=True)
    identifiers = np.concatenate(id_parts)
    del id_parts
    identifiers.sort()
    repeated = identifiers[1:][identifiers[1:] == identifiers[:-1]]
    duplicate_ids = np.unique(repeated)
    issues['duplicate_ride_id_extra_rows'] = int(len(repeated))
    issues['distinct_duplicated_ride_ids'] = int(len(duplicate_ids))
    duplicate_sample = [f'{int(value):016X}' for value in duplicate_ids[:20]]
    daily = pd.Series(day_counts, name='trips').sort_index()
    daily.index = pd.DatetimeIndex(daily.index, name='date')
    expected_dates = pd.date_range(expected_start, expected_end - pd.Timedelta(days=1), freq='D', name='date')
    report = {
        'audit_version': AUDIT_VERSION, 'year': year, 'rows': total,
        'counted_rides': int(daily.sum()), 'csv_members': files,
        'first_started_at': str(minimum_start), 'last_started_at': str(maximum_start),
        'expected_days': len(expected_dates), 'observed_days': len(daily),
        'missing_dates': [date.strftime('%Y-%m-%d') for date in expected_dates.difference(daily.index)],
        'schema_variants': [{'columns': list(cols), 'members': sorted(names)} for cols, names in schemas.items()],
        'missing_values': _json_counts(missing), 'issues': _json_counts(issues),
        'duplicate_id_sample': duplicate_sample, 'rides_by_hour': _json_counts(hour_counts),
        'outside_year_sample': outside_samples,
        'categories': {name: _json_counts(values) for name, values in categories.items()},
        'time_policy': 'Aggregate printed dates as NYC local clock labels; no UTC conversion. Source has no offsets.',
        'duration_policy': 'Flag duration anomalies without dropping rides: target counts recorded starts, not valid durations.',
        'scope_policy': 'Keep rides whose printed start dates fall within the selected year; explicitly exclude other years, preserving raw data.',
    }
    return daily.to_frame(), report


def acquire_and_audit(reuse_audit=True):
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    downloads = [download_archive(item['url'], item.get('sha256')) for item in config['archives']]
    sources = [path for path, _ in downloads]
    metadata = [item for _, item in downloads]
    daily_path = PROCESSED_DIR / 'citibike_daily.csv'
    report_path = REPORT_DIR / 'audit.json'
    if reuse_audit and report_path.exists() and daily_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if (report.get('audit_version') == AUDIT_VERSION and report.get('year') == config['year']
                and [item['sha256'] for item in report.get('sources', [])] == [item['sha256'] for item in metadata]
                and report.get('daily_sha256') == sha256_file(daily_path)):
            print('Using matching audited table; reuse_audit=False repeats the full row audit.', flush=True)
            return pd.read_csv(daily_path, index_col='date', parse_dates=['date']), report
    daily, report = audit_archive(sources, config['year'])
    report['sources'] = metadata
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if report['missing_dates'] or report['issues']['unparsed_start']:
        raise ValueError('Coverage/timestamp issues require investigation; see reports/citibike/audit.json.')
    if report['issues']['duplicate_ride_id_extra_rows']:
        raise ValueError('Duplicate ride IDs require investigation before accepting daily counts; see audit.json.')
    if int(daily['trips'].sum()) + report['issues']['outside_selected_year'] != report['rows']:
        raise ValueError('Daily counts plus explicit scope exclusions do not reconcile with raw rows.')
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_path, date_format='%Y-%m-%d')
    report['daily_sha256'] = sha256_file(daily_path)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return daily, report


def one_step_features(daily):
    """Predict day t using only counts observed through t-1."""
    frame = daily.sort_index().copy()
    frame['weekday'] = frame.index.dayofweek
    frame['month'] = frame.index.month
    frame['lag_1'] = frame['trips'].shift(1)
    frame['lag_7'] = frame['trips'].shift(7)
    frame['past_7_day_mean'] = frame['trips'].shift(1).rolling(7).mean()
    return frame.dropna()


if __name__ == '__main__':
    daily, audit = acquire_and_audit()
    print(f"Accepted {audit['counted_rides']:,} rides across {len(daily)} days; "
          f"excluded {audit['issues']['outside_selected_year']} starts outside {audit['year']}.")
