"""Random real training-side inputs for the demo, without labels."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from secrets import choice

ROOT = Path(__file__).resolve().parents[1]
WEATHER_DAYS = {'counts': 0, 'weather_cpu': 8, 'weather_gru': 1}


@lru_cache(maxsize=2)
def pool(kind):
    return json.loads((ROOT / 'examples' / f'random_{kind}.json').read_text())


def random_example(kind, model='weather_gru', exclude=None):
    if kind not in ('mushrooms', 'bikes'):
        raise ValueError('Unknown example type.')
    if kind == 'bikes' and model not in WEATHER_DAYS:
        raise ValueError('Unknown bike model.')
    selected = choice([row for row in pool(kind) if row['example_id'] != exclude])
    result = deepcopy(selected)
    if kind == 'bikes':
        days = WEATHER_DAYS[model]
        result['input']['model'] = model
        result['input']['forecasts'] = result['input']['forecasts'][-days:] if days else []
    return result
