"""Reproduce merged638-day forecast feature tables from preserved source tables."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit/weather"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = []
    for stem in ["temperature", "jma_precip_wind"]:
        paths = [OUT / f"forecast_daily_{stem}_2022.csv", OUT / f"forecast_daily_{stem}.csv"]
        tables = [pd.read_csv(path, parse_dates=["date"]).set_index("date") for path in paths]
        joined = pd.concat(tables)
        assert joined.index.equals(pd.date_range("2022-01-01", "2023-09-30"))
        assert joined.notna().all().all()
        output = OUT / f"forecast_daily_{stem}_expanded.csv"
        joined.to_csv(output, float_format="%.6f", index_label="date")
        records.append({"file": str(output), "days": len(joined), "sha256": sha256(output),
                        "sources": [{"file": path.name, "sha256": sha256(path)} for path in paths],
                        "forecast_lead_primary": 48})
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "derived_files": records,
                "no_trip_labels_read": True, "primary_day2_contract": True,
                "code_sha256": sha256(Path(__file__))}
    (OUT / "expanded_weather_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
