"""Load only explicitly allowed development partitions."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_mushrooms(partition='search', allow_confirmation=False):
    if partition == 'confirmation' and not allow_confirmation:
        raise RuntimeError('Freeze finalists and request root review before confirmation scoring.')
    manifest = json.loads((ROOT / 'common/mushroom_partitions.json').read_text())
    if partition not in manifest['partitions']:
        raise ValueError('Unknown research partition.')
    frame = pd.read_csv(ROOT / 'data/mushroom_development.csv', index_col='source_row_id')
    result = frame.loc[manifest['partitions'][partition]].copy()
    assert len(result) == manifest['sizes'][partition]
    return result


def load_citibike_training_history():
    daily = pd.read_csv(ROOT / 'data/citibike_development_daily.csv', index_col='date', parse_dates=['date'])
    assert daily.index.max() == pd.Timestamp('2023-09-30')
    return daily.loc[:'2023-06-30'].copy()
