"""Cache public, archived GFS forecasts, without any trip target labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit" / "weather"
ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLES = ["temperature_2m_previous_day1", "temperature_2m_previous_day2"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(start: str, end: str, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    padded_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    params = {
        "latitude": "40.7128", "longitude": "-74.0060",
        "start_date": start, "end_date": padded_end,
        "hourly": ",".join(VARIABLES), "models": "gfs_global",
        "timezone": "America/New_York", "timeformat": "unixtime",
        "temperature_unit": "celsius",
    }
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    raw = OUT / f"{name}_raw.json"
    metadata = {
        "endpoint": ENDPOINT, "request_url": url, "parameters": params,
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "included_local_date_range": {"start": start, "end": end},
        "boundary_padding_policy": "Request one extra end date because the API uses a fixed current UTC offset. Trim to the explicitly permitted Jan-Sep local dates after Unix timestamp conversion.",
        "documentation": "https://open-meteo.com/en/docs/previous-runs-api",
        "timezone_documentation": "https://open-meteo.com/en/docs",
        "license_page": "https://open-meteo.com/en/licence",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Weather forecasts by Open-Meteo.com, sourced from NOAA GFS; transformed into daily NYC temperature summaries.",
        "no_paid_endpoint_or_key": True,
        "forecast_leads": {"previous_day1": "24h before each valid hour; optional research ablation",
                           "previous_day2": "48h before each valid hour; primary conservative research input"},
        "availability_assumption": "Archive is an initialization-time proxy, not historical API publication timestamps. 48h lead leaves at least24h before the midnight prediction cutoff even on a25h NYC fall-DST day. Actual publication availability cannot be proved from the response alone.",
        "data_contract": "Forecast-assisted models require these forecast inputs at inference. This is separate from the existing counts-only API.",
    }
    existing = OUT / f"{name}_metadata.json"
    reusable = raw.exists() and existing.exists() and json.loads(existing.read_text()).get("request_url") == url
    if not reusable:
        request = urllib.request.Request(url, headers={"User-Agent": "CloudAI-student-research/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
                metadata["http_status"] = response.status
                metadata["response_headers"] = {key: response.headers.get(key) for key in
                    ["Content-Type", "Date", "ETag", "Last-Modified"]}
        except urllib.error.HTTPError as error:
            failure = error.read().decode("utf-8", errors="replace")
            metadata.update(http_status=error.code, error_response=failure, download_success=False)
            (OUT / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"Public weather source unavailable: HTTP{error.code}: {failure[:400]}") from error
        raw.write_bytes(data)
        (OUT / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    else:
        existing = OUT / f"{name}_metadata.json"
        if existing.exists():
            metadata = json.loads(existing.read_text())
        metadata["cached_response_reused"] = True
    response = json.loads(raw.read_text())
    if "error" in response:
        raise RuntimeError(response)
    hourly = response["hourly"]
    utc_index = pd.to_datetime(hourly["time"], unit="s", utc=True)
    if utc_index.has_duplicates or not utc_index.is_monotonic_increasing:
        raise RuntimeError("Weather hours are not sorted and unique.")
    local_index = utc_index.tz_convert("America/New_York")
    frame = pd.DataFrame({variable: hourly[variable] for variable in VARIABLES}, index=local_index)
    start_local = pd.Timestamp(start, tz="America/New_York")
    stop_local = pd.Timestamp(end, tz="America/New_York") + pd.DateOffset(days=1)
    trimmed = frame.loc[(frame.index >= start_local) & (frame.index < stop_local)].copy()
    expected = pd.date_range(start_local, stop_local, freq="h", inclusive="left")
    missing_hours = expected.difference(trimmed.index)
    if len(missing_hours):
        raise RuntimeError(f"Missing{len(missing_hours)} real NYC hours; do not silently fill.")
    if trimmed.isna().any().any():
        nulls = trimmed.isna().sum().to_dict()
        raise RuntimeError(f"Archived forecasts contain unavailable values: {nulls}")
    if not np.isfinite(trimmed.to_numpy()).all():
        raise RuntimeError("Nonfinite archived forecasts.")
    dates = trimmed.index.tz_localize(None).normalize()
    grouped = trimmed.groupby(dates)
    daily = grouped.agg(["mean", "min", "max"])
    daily.columns = [f"forecast_temp_day{1 if 'day1' in variable else 2}_{statistic}"
                     for variable, statistic in daily.columns]
    daily.index.name = "date"
    daily_name = ("forecast_daily_temperature.csv" if name == "gfs_jan_sep_2023" else
                  "forecast_daily_temperature_2022.csv" if name == "gfs_2022" else f"{name}_daily_temperature.csv")
    hourly_name = ("forecast_hourly_temperature.csv" if name == "gfs_jan_sep_2023" else
                   "forecast_hourly_temperature_2022.csv" if name == "gfs_2022" else f"{name}_hourly_temperature.csv")
    daily.to_csv(OUT / daily_name, float_format="%.6f", index=True)
    trimmed.assign(utc_timestamp=trimmed.index.tz_convert("UTC").astype(str)).to_csv(
        OUT / hourly_name, index_label="local_timestamp")
    hour_counts = grouped.size()
    metadata.update({
        "download_success": True,
        "raw_response_sha256": digest(raw), "raw_response_bytes": raw.stat().st_size,
        "daily_features_file": daily_name,
        "hourly_features_file": hourly_name,
        "daily_features_sha256": digest(OUT / daily_name),
        "hourly_features_sha256": digest(OUT / hourly_name),
        "response_summary": {key: response.get(key) for key in
            ["latitude", "longitude", "elevation", "timezone", "timezone_abbreviation", "utc_offset_seconds", "hourly_units"]},
        "raw_hours": len(frame), "included_real_local_hours": len(trimmed), "included_days": len(daily),
        "hour_counts_distribution": hour_counts.value_counts().sort_index().to_dict(),
        "non_24_hour_days": {date.strftime("%Y-%m-%d"): int(count) for date, count in hour_counts.items() if count != 24},
        "excluded_boundary_hours": len(frame) - len(trimmed),
        "missing_hours": 0, "missing_forecast_values": 0,
        "dst_policy": "Request Unix UTC timestamps, convert each instant with America/New_York timezone rules, then aggregate by actual local date. Never reuse the response fixed UTC offset across the date range.",
        "aggregation": "Mean/min/max of hourly temperature forecasts for each true local calendar date, preserving23h/25h DST days.",
        "mushroom_or_bike_labels_read": False,
        "reserved_original_test_targets_read": False,
    })
    (OUT / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ["download_success", "included_days", "included_real_local_hours", "non_24_hour_days", "excluded_boundary_hours", "daily_features_sha256"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--year", type=int, choices=[2022, 2023], default=2023)
    args = parser.parse_args()
    if args.year == 2022:
        if args.probe:
            raise ValueError("Only the approved bounded full2022request is supported.")
        download("2022-01-01", "2022-12-31", "gfs_2022")
    else:
        download("2023-01-01", "2023-01-03" if args.probe else "2023-09-30",
                 "gfs_probe" if args.probe else "gfs_jan_sep_2023")
