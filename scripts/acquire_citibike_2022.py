"""Audit one extra past training year from the exact official annual archive."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
import re
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/citibike2022"
OUT = ROOT / "audit/citibike2022"
SOURCE = "https://s3.amazonaws.com/tripdata/2022-citibike-tripdata.zip"
EXPECTED_BYTES = 1231367136
EXISTING_2023 = ROOT / "data/raw/citibike/2023-citibike-tripdata.zip"
EXPECTED_2023_SHA = "f5ac8d2b9556a4f18925569681d60e960eae44ce7e2999c74726c8e1ff706c07"
CHUNK = 200000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    final = RAW / "2022-citibike-tripdata.zip"
    meta_path = RAW / "2022-citibike-tripdata.download.json"
    if final.exists():
        metadata = json.loads(meta_path.read_text())
        assert final.stat().st_size == EXPECTED_BYTES
        assert sha256(final) == metadata["sha256"]
        return final, metadata
    partial = final.with_suffix(".zip.part")
    digest = hashlib.sha256()
    written, last_report = 0, 0
    with urllib.request.urlopen(SOURCE, timeout=120) as response, partial.open("wb") as stream:
        length = int(response.headers.get("Content-Length", 0))
        if length != EXPECTED_BYTES or length > 3 * 1024**3:
            raise RuntimeError(f"Unexpected/oversized source{length}; stop for root review.")
        metadata = {"source": SOURCE, "discovery_url": "https://s3.amazonaws.com/tripdata?prefix=2022&max-keys=1000",
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "http_length": length, "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "license_source": "https://citibikenyc.com/system-data"}
        for block in iter(lambda: response.read(8 * 1024**2), b""):
            stream.write(block)
            digest.update(block)
            written += len(block)
            if written - last_report >= 128 * 1024**2:
                print(f"2022download {written / 1024**2:.0f}/{length / 1024**2:.0f}MiB", flush=True)
                last_report = written
    if written != EXPECTED_BYTES or not zipfile.is_zipfile(partial):
        raise RuntimeError("Incomplete or invalid official archive.")
    metadata.update(bytes=written, sha256=digest.hexdigest())
    partial.replace(final)
    dump(meta_path, metadata)
    return final, metadata


def archive_members(path, january_only=False):
    # Read-only reuse of the audited recursive ZIP reader. This helper does not
    # call acquisition, mutate repo paths or read any dataset on import.
    spec = importlib.util.spec_from_file_location("citibike_archive_reader", ROOT / "src/citibike.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with zipfile.ZipFile(path) as outer:
        if not january_only:
            yield from module._csv_members(outer, path.name + "!/")
        else:
            # Restrict the existing2023archive to its January inner ZIP. Never
            # iterate through or count October-December2023 target dates.
            january = [info for info in outer.infolist() if re.search(r"(^|/)202301-citibike-tripdata\.zip$", info.filename)]
            if len(january) != 1:
                raise RuntimeError("Expected exactly one January2023 nested archive.")
            import tempfile
            with tempfile.TemporaryFile() as temp, outer.open(january[0]) as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    temp.write(block)
                temp.seek(0)
                with zipfile.ZipFile(temp) as child:
                    yield from module._csv_members(child, path.name + "!/" + january[0].filename + "!/")


def audit(path, metadata):
    counts, issues = Counter(), Counter()
    files, parts, id_schemas = [], [], set()
    start, end = pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")
    for origin, archive, january_only in [("annual2022", path, False), ("boundaryJanuary2023", EXISTING_2023, True)]:
        if january_only:
            print("Verifying existing2023archive hash without reading model targets...", flush=True)
            if sha256(archive) != EXPECTED_2023_SHA:
                raise RuntimeError("Existing2023archive fingerprint changed.")
        for member, info, stream in archive_members(archive, january_only=january_only):
            print("Auditing " + member, flush=True)
            n, kept, member_issues = 0, 0, Counter()
            schema = None
            for chunk in pd.read_csv(stream, usecols=["ride_id", "started_at"], dtype="string", chunksize=CHUNK):
                schema = tuple(chunk.columns)
                id_schemas.add(schema)
                timestamps = pd.to_datetime(chunk.started_at, format="mixed", errors="coerce")
                if timestamps.dt.tz is not None:
                    raise RuntimeError("Unexpected timezone offsets; explicit policy needed.")
                valid = timestamps.ge(start) & timestamps.lt(end)
                member_issues["unparsed_start"] += int(timestamps.isna().sum())
                member_issues["excluded_other_year_rows"] += int((timestamps.notna() & ~valid).sum())
                if not january_only and timestamps.isna().any():
                    raise RuntimeError("Unparsed2022start times require review.")
                selected = chunk.loc[valid]
                ids = selected.ride_id
                if not ids.str.fullmatch(r"[0-9A-Fa-f]{16}", na=False).all():
                    raise RuntimeError("Invalid scoped ride identifier.")
                id_array = np.fromiter((int(value, 16) for value in ids), dtype=np.uint64, count=len(ids))
                parts.append(id_array)
                days = timestamps.loc[valid].dt.normalize()
                counts.update(days.value_counts().to_dict())
                n += len(chunk)
                kept += int(valid.sum())
            files.append({"origin": origin, "member": member, "bytes_uncompressed": info.file_size,
                          "crc32": f"{info.CRC:08x}", "read_rows": n, "scoped_2022_rides": kept,
                          "issues": dict(member_issues), "read_schema": list(schema)})
            issues.update(member_issues)
            print(f"  scanned{n:,}; retained{kept:,}starts in2022", flush=True)
    print("Sorting scoped rideIDs for exact cross-file duplicate audit...", flush=True)
    ids = np.concatenate(parts)
    del parts
    ids.sort()
    repeated = ids[1:] == ids[:-1]
    issues["duplicate_scoped_ride_id_extra_rows"] = int(repeated.sum())
    if repeated.any():
        raise RuntimeError("Duplicate2022rideIDs require correction before accepting counts.")
    daily = pd.Series(counts, name="trips").sort_index().to_frame()
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    expected = pd.date_range(start, end - pd.Timedelta(days=1), name="date")
    missing = expected.difference(daily.index)
    if len(missing) or not daily.index.equals(expected):
        raise RuntimeError(f"2022coverage incomplete: {len(missing)}missingdays.")
    if daily.trips.le(0).any():
        raise RuntimeError("Nonpositive2022daily counts require review.")
    final = ROOT / "data/citibike_2022_daily.csv"
    daily.to_csv(final, date_format="%Y-%m-%d")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "year": 2022,
        "annual_source": metadata, "boundary_source": {"path_read_only": str(EXISTING_2023), "sha256": EXPECTED_2023_SHA,
            "members_allowed": "January2023innerarchiveonly; retain starts in2022"},
        "daily_file": str(final), "daily_sha256": sha256(final), "days": len(daily),
        "daily_min": int(daily.trips.min()), "daily_max": int(daily.trips.max()), "counted_rides": int(daily.trips.sum()),
        "boundary_carryover_rides_from_january2023": sum(item["scoped_2022_rides"] for item in files if item["origin"] == "boundaryJanuary2023"),
        "files": files, "read_schemas": [list(schema) for schema in id_schemas], "issues": dict(issues),
        "missing_dates": [], "reserved2023OctDectargets_not_read": True,
        "target_policy": "All recorded ride starts by printed NYC local calendar dates; no duration/station-null filtering.",
        "scope": "Optional additional past-year training experiment. Base2023question and existing repository remain unchanged.",
        "split_requirements": "Keep original2023April29-June30 chronological tuning windows and July-September frozen finalist comparison. All extra2022targets precede these windows.",
        "null_policy": "No missing start/IDs permitted for scoped rides. Unused station fields are not needed for citywide ride starts.",
    }
    dump(OUT / "audit.json", report)
    print(json.dumps({key: report[key] for key in ["days", "counted_rides", "boundary_carryover_rides_from_january2023", "daily_sha256"]}, indent=2), flush=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not EXISTING_2023.exists():
        sys.path.insert(0, str(ROOT / 'src'))
        from citibike import download_archive
        download_archive('https://s3.amazonaws.com/tripdata/2023-citibike-tripdata.zip', EXPECTED_2023_SHA)
    path, metadata = download()
    with threadpool_limits(limits=2):
        audit(path, metadata)


if __name__ == "__main__":
    main()
