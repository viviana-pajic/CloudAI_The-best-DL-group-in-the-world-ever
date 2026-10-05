"""Citi Bike acquisition and daily features; validate on real archives before use."""
from pathlib import Path
import re
import urllib.request
import zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def download_and_aggregate(url):
    """Read official archive CSVs in chunks, retaining missing-coverage checks."""
    if not url.startswith('https://s3.amazonaws.com/tripdata/'):
        raise ValueError('Use an official tripdata bucket HTTPS URL.')
    filename=url.rsplit('/',1)[-1]
    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.zip',filename):
        raise ValueError('Expected a ZIP filename.')
    path=ROOT/'data/raw/citibike'/filename
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url,timeout=120) as response, path.open('wb') as out:
            while chunk:=response.read(1024*1024):
                out.write(chunk)
    totals=[]
    with zipfile.ZipFile(path) as archive:
        files=[n for n in archive.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/')]
        if not files:
            raise ValueError('Archive has no direct CSV members. Check archive structure.')
        for member in files:
            # Stream chunks without extracting paths into the filesystem.
            with archive.open(member) as stream:
                for chunk in pd.read_csv(stream,chunksize=100000,low_memory=False):
                    col=next((c for c in ['started_at','starttime','Start Time'] if c in chunk),None)
                    if col is None:
                        raise ValueError('Unknown timestamp schema; document and map it explicitly.')
                    dates=pd.to_datetime(chunk[col],errors='coerce')
                    if dates.isna().any():
                        raise ValueError('Unparsed timestamps found. Inspect them before aggregation.')
                    totals.append(dates.dt.normalize().value_counts().rename('trips'))
    daily=pd.concat(totals).groupby(level=0).sum().sort_index()
    # No silent zero-fill: a gap could mean missing source coverage.
    full_index=pd.date_range(daily.index.min(),daily.index.max(),freq='D')
    daily=daily.reindex(full_index)
    if daily.isna().any():
        raise ValueError('Missing days. Confirm coverage before deciding whether a zero is justified.')
    daily.index.name='date'
    return daily.rename('trips').to_frame()


def one_step_features(daily):
    """Each row predicts day t after the actual count through t-1 is known."""
    frame=daily.sort_index().copy()
    frame['weekday']=frame.index.dayofweek
    frame['month']=frame.index.month
    frame['lag_1']=frame['trips'].shift(1)
    frame['lag_7']=frame['trips'].shift(7)
    frame['past_7_day_mean']=frame['trips'].shift(1).rolling(7).mean()
    return frame.dropna()
