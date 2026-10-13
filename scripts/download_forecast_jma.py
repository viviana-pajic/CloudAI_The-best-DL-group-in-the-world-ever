"""Public JMA GSM archived precipitation/wind forecasts at48h lead only."""
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
OUT = ROOT / "audit/weather"
ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
PRECIP = "precipitation_previous_day2"
WIND = "wind_speed_10m_previous_day2"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(probe=False, year=2023, start_date=None, end_date=None, run_name=None):
    OUT.mkdir(exist_ok=True)
    start = f"{year}-01-01"
    end = "2023-01-03" if probe else "2023-09-30" if year == 2023 else "2022-12-31"
    name = "jma_probe" if probe else "jma_jan_sep_2023" if year == 2023 else "jma_2022"
    start = start_date or start
    end = end_date or end
    name = run_name or name
    params = {
        "latitude": "40.7128", "longitude": "-74.0060", "start_date": start,
        "end_date": (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "models": "jma_gsm", "hourly": PRECIP + "," + WIND,
        "timezone": "America/New_York", "timeformat": "unixtime",
        "wind_speed_unit": "kmh", "precipitation_unit": "mm",
    }
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    raw = OUT / f"{name}_raw.json"
    meta_path = OUT / f"{name}_metadata.json"
    meta = {
        "request_url": url, "parameters": params, "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "included_local_date_range": {"start": start, "end": end},
        "documentation": "https://open-meteo.com/en/docs/previous-runs-api",
        "model_documentation": "https://open-meteo.com/en/docs/jma-api",
        "parameter_documentation": "https://open-meteo.com/en/docs",
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_page": "https://open-meteo.com/en/licence",
        "attribution": "Weather forecasts by Open-Meteo.com, sourced from JMA GSM; transformed into daily NYC precipitation/wind summaries.",
        "no_paid_endpoint_or_key": True,
        "forecasts_only_no_observations_or_reanalysis": True,
        "forecast_lead_hours": 48,
        "publication_latency_caveat": "Lead offset is an initialization proxy. Actual historical API publication timestamps are unavailable. The final precipitation hour ends at the next midnight: a48h lead leaves a24h initialization buffer on normal days and23h on a25h fall-DST day. Actual API publication availability still needs validation.",
        "native_model_resolution": "JMA GSM global0.5degree(~55km), six-hourly fields interpolated to hourly values by Open-Meteo.",
        "inference_contract": "New forecast-assisted input dependency, separate from counts-only app/model.",
    }
    if raw.exists() and meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous.get("request_url") != url:
            raise RuntimeError("Cached request differs.")
        meta = previous
        meta["cached_response_reused"] = True
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "CloudAI-student-research/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                data = response.read()
                meta.update(http_status=response.status, response_headers={key: response.headers.get(key) for key in ["Content-Type", "Date", "ETag", "Last-Modified"]})
            raw.write_bytes(data)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            meta.update(http_status=error.code, error_response=body, download_success=False)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"JMA forecast archive unavailable HTTP{error.code}: {body[:400]}") from error
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    response = json.loads(raw.read_text())
    times = pd.to_datetime(response["hourly"]["time"], unit="s", utc=True).tz_convert("America/New_York")
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise RuntimeError("Forecast timestamps not sorted/unique.")
    hourly = pd.DataFrame({PRECIP: response["hourly"][PRECIP], WIND: response["hourly"][WIND]}, index=times)
    start_local = pd.Timestamp(start, tz="America/New_York")
    stop_local = pd.Timestamp(end, tz="America/New_York") + pd.DateOffset(days=1)
    expected_wind = pd.date_range(start_local, stop_local, inclusive="left", freq="h")
    expected_precip = expected_wind + pd.Timedelta(hours=1)
    for variable, expected in [(WIND, expected_wind), (PRECIP, expected_precip)]:
        if len(expected.difference(times)):
            raise RuntimeError(f"Incomplete valid hours for{variable}.")
        values = hourly.loc[expected, variable]
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            meta.update(download_success=False, unavailable_variable=variable, missing_values=int(values.isna().sum()))
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"Unavailable archived{variable}: {int(values.isna().sum())}missingvalues.")
        if (values < 0).any() and (variable != PRECIP or values.min() < -0.100001):
            raise RuntimeError(f"Negative{variable}values exceed the authorized0.1mm precipitation cleanup rule.")
    wind = hourly.loc[expected_wind, WIND]
    precip = hourly.loc[expected_precip, PRECIP]
    wind_dates = wind.index.tz_localize(None).normalize()
    # Accumulations belong to the preceding real hour. Shift valid timestamps
    # back by one elapsed hour before deriving the local date.
    precip_dates = (precip.index - pd.Timedelta(hours=1)).tz_localize(None).normalize()
    negative_forecasts = precip.lt(0)
    clean_precip = precip.clip(lower=0)
    daily = pd.DataFrame({
        "forecast_jma_day2_precip_sum": clean_precip.groupby(precip_dates).sum(),
        "forecast_jma_day2_precip_raw_sum": precip.groupby(precip_dates).sum(),
        "forecast_jma_day2_precip_negative_hours": negative_forecasts.groupby(precip_dates).sum(),
        "forecast_jma_day2_wind_mean": wind.groupby(wind_dates).mean(),
        "forecast_jma_day2_wind_max": wind.groupby(wind_dates).max(),
    })
    daily.index.name = "date"
    assert daily.index.equals(pd.date_range(start, end))
    daily_name = f"{name}_daily.csv" if probe else "forecast_daily_jma_precip_wind.csv" if year == 2023 else "forecast_daily_jma_precip_wind_2022.csv"
    if run_name:
        daily_name = f"{name}_daily.csv"
    hourly_name = f"{name}_hourly.csv"
    daily.to_csv(OUT / daily_name, float_format="%.6f")
    hourly.loc[expected_wind.union(expected_precip)].to_csv(OUT / hourly_name, index_label="local_valid_timestamp")
    counts = wind.groupby(wind_dates).size()
    meta.update({
        "download_success": True, "raw_response_sha256": sha256(raw), "raw_response_bytes": raw.stat().st_size,
        "daily_features_file": daily_name, "daily_features_sha256": sha256(OUT / daily_name),
        "hourly_features_file": hourly_name, "hourly_features_sha256": sha256(OUT / hourly_name),
        "included_days": len(daily), "actual_local_hours": len(expected_wind),
        "hour_counts_distribution": counts.value_counts().sort_index().to_dict(),
        "non24hour_days": {date.strftime("%Y-%m-%d"): int(count) for date, count in counts.items() if count != 24},
        "hourly_units": response["hourly_units"],
        "response_coordinates": {key: response.get(key) for key in ["latitude", "longitude", "elevation", "timezone", "utc_offset_seconds"]},
        "precipitation_aggregation": "Sum of preceding-hour forecasts with valid instants01:00 through next00:00 assigned by interval start date. Preserve real23h/25h DST days.",
        "precipitation_cleanup_rule": "Negative forecast precipitation with magnitude<=0.1mm is physically invalid and floored to0. Raw response/hourly values/rawdaily sums are preserved; negative-hour count is a separate ablation feature. Larger negative values fail. Chosen before any bike validation labels.",
        "negative_precipitation_hours": int(negative_forecasts.sum()),
        "negative_precipitation_original_sum_mm": float(precip.loc[negative_forecasts].sum()),
        "negative_precipitation_dates": {date.strftime("%Y-%m-%d"): int(count) for date, count in negative_forecasts.groupby(precip_dates).sum().items() if count},
        "wind_aggregation": "Mean/max of instantaneous forecasts from real00:00 through23:00 local hours, preserving23h/25h DST days.",
        "dst_handling": "Convert Unix UTC to trueAmerica/New_York rules rather than API fixedutc_offset_seconds.",
        "no_trip_targets_read": True, "missing_forecast_values": 0,
    })
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: meta[key] for key in ["download_success", "included_days", "actual_local_hours", "non24hour_days", "hourly_units", "daily_features_sha256"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--year", type=int, choices=[2022, 2023], default=2023)
    args = parser.parse_args()
    if args.year == 2022 and args.probe:
        raise ValueError("Only the approved bounded full2022request is supported.")
    download(args.probe, args.year)
