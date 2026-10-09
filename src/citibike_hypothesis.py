"""Within-week demand contrast and dependence-aware exploratory inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / 'citibike/hypothesis_plan.json'


def complete_week_contrasts(daily, start, end, holdout_start):
    if pd.Timestamp(end) >= pd.Timestamp(holdout_start):
        raise ValueError('Analysis must end before the candidate final-test period.')
    if not isinstance(daily.index, pd.DatetimeIndex) or daily.index.tz is not None:
        raise ValueError('Expected printed local calendar dates in a naive DatetimeIndex.')
    selected = daily.loc[(daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end)), ['trips']].sort_index().copy()
    expected = pd.date_range(start, end, freq='D')
    if selected.index.has_duplicates or not selected.index.equals(expected):
        raise ValueError('Development dates must be unique, consecutive and cover the planned period.')
    values = selected['trips'].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Weekly normalization requires finite positive daily counts.')
    selected['weekday'] = selected.index.dayofweek
    selected['week_start'] = selected.index - pd.to_timedelta(selected['weekday'], unit='D')
    weeks = selected.pivot(index='week_start', columns='weekday', values='trips').reindex(columns=range(7))
    complete = weeks.dropna()
    result = pd.DataFrame(index=complete.index)
    result['weekday_mean'] = complete.loc[:, 0:4].mean(axis=1)
    result['weekend_mean'] = complete.loc[:, 5:6].mean(axis=1)
    result['week_mean'] = complete.mean(axis=1)
    result['difference_rides'] = result['weekday_mean'] - result['weekend_mean']
    result['normalized_contrast'] = result['difference_rides'] / result['week_mean']
    included = selected['week_start'].isin(result.index)
    excluded = [date.strftime('%Y-%m-%d') for date in selected.index[~included]]
    return result, excluded


def stationary_indices(n, repetitions, expected_block_length, seed=42):
    """Continue the next observation or restart uniformly; runs are geometric."""
    if n < 2 or repetitions < 1 or not 1 <= expected_block_length <= n:
        raise ValueError('Invalid bootstrap size, repetition count or block length.')
    rng = np.random.default_rng(seed)
    indices = np.empty((repetitions, n), dtype=np.int64)
    indices[:, 0] = rng.integers(n, size=repetitions)
    restart_probability = 1 / expected_block_length
    for position in range(1, n):
        restart = rng.random(repetitions) < restart_probability
        fresh = rng.integers(n, size=repetitions)
        indices[:, position] = np.where(restart, fresh, (indices[:, position - 1] + 1) % n)
    return indices


def bootstrap_mean(values, block_weeks, repetitions=20_000, seed=42, confidence=0.95):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Expected one finite weekly contrast per observation.')
    if len(values) < 2 * block_weeks:
        raise ValueError('Too few weeks for the chosen block-length sensitivity check.')
    if not 0 < confidence < 1:
        raise ValueError('Confidence level must be between zero and one.')
    indices = stationary_indices(len(values), repetitions, block_weeks, seed)
    draws = values[indices].mean(axis=1)
    point = float(values.mean())
    tail = (1 - confidence) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    errors = draws - point
    extreme_replicates = int((np.abs(errors) >= abs(point)).sum())
    p_value = (1 + extreme_replicates) / (repetitions + 1)
    result = {
        'expected_block_weeks': block_weeks,
        'estimate': point,
        'ci_lower': float(2 * point - high),
        'ci_upper': float(2 * point - low),
        'bootstrap_standard_error': float(draws.std(ddof=1)),
        'approximate_two_sided_p': p_value,
        'null_extreme_replicates': extreme_replicates,
        'monte_carlo_resolution': 1 / (repetitions + 1),
    }
    return result, draws


def analyse(daily, plan=None):
    if plan is None:
        plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
    weeks, excluded = complete_week_contrasts(
        daily, plan['analysis_start'], plan['analysis_end'], plan['candidate_holdout_start']
    )
    values = weeks['normalized_contrast'].to_numpy()
    sensitivity = []
    primary_draws = None
    for block in plan['sensitivity_expected_block_weeks']:
        result, draws = bootstrap_mean(values, block, plan['bootstrap_replicates'], plan['seed'], plan['confidence_level'])
        sensitivity.append(result)
        if block == plan['primary_expected_block_weeks']:
            primary = result
            primary_draws = draws
    if primary_draws is None:
        raise ValueError('Primary block length must also be in the sensitivity list.')
    stable_positive = all(item['ci_lower'] > 0 for item in sensitivity)
    stable_negative = all(item['ci_upper'] < 0 for item in sensitivity)
    primary_excludes_zero = primary['ci_lower'] > 0 or primary['ci_upper'] < 0
    supports_calendar_candidate = primary_excludes_zero and (stable_positive or stable_negative)
    quarters = []
    for quarter, frame in weeks.groupby(weeks.index.quarter):
        quarters.append({'quarter_by_week_start': int(quarter), 'weeks': len(frame),
                         'mean_normalized_contrast': float(frame['normalized_contrast'].mean())})
    autocorrelations = {}
    for lag in range(1, 5):
        earlier, later = values[:-lag], values[lag:]
        autocorrelations[str(lag)] = (None if np.ptp(earlier) == 0 or np.ptp(later) == 0
                                     else float(weeks['normalized_contrast'].autocorr(lag)))
    summary = {
        'analysis_date': plan['analysis_date'], 'analysis_type': plan['status'],
        'planned_start': plan['analysis_start'], 'planned_end': plan['analysis_end'],
        'candidate_holdout_start': plan['candidate_holdout_start'],
        'complete_weeks': len(weeks), 'included_days': len(weeks) * 7,
        'first_included_date': weeks.index.min().strftime('%Y-%m-%d'),
        'last_included_date': (weeks.index.max() + pd.Timedelta(days=6)).strftime('%Y-%m-%d'),
        'excluded_incomplete_week_dates': excluded,
        'mean_weekday_rides': float(weeks['weekday_mean'].mean()),
        'mean_weekend_rides': float(weeks['weekend_mean'].mean()),
        'mean_paired_difference_rides': float(weeks['difference_rides'].mean()),
        'positive_weeks': int((values > 0).sum()),
        'negative_weeks': int((values < 0).sum()),
        'primary': primary, 'sensitivity': sensitivity, 'quarter_descriptives': quarters,
        'weekly_contrast_autocorrelations': autocorrelations,
        'confidence_level': plan['confidence_level'], 'replicates': plan['bootstrap_replicates'], 'seed': plan['seed'],
        'primary_interval_excludes_zero': primary_excludes_zero,
        'stable_direction_across_block_lengths': stable_positive or stable_negative,
        'continue_forecast_candidate': supports_calendar_candidate,
        'forecast_accuracy_evaluated': False, 'candidate_final_test_used': False,
        'decision': ('Continue the proposed next-day recorded ride-count task; include calendar weekday as a candidate input. '
                     'Chronological baseline/model evaluation and team scope agreement remain required.' if supports_calendar_candidate else
                     'This contrast does not establish a stable weekly difference. Reconsider or gather more development evidence before relying on weekday.'),
    }
    return weeks, summary, primary_draws


if __name__ == '__main__':
    data_path = ROOT / 'data/processed/citibike_daily.csv'
    manifest = json.loads((ROOT / 'citibike/archive_manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest['daily_sha256']:
        raise ValueError('Daily counts do not match the audited source manifest; rerun acquisition.')
    daily = pd.read_csv(data_path, index_col='date', parse_dates=['date'])
    weeks, summary, _ = analyse(daily)
    (ROOT / 'citibike/hypothesis_results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
