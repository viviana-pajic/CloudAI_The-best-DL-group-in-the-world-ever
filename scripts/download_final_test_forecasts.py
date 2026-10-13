"""Get reserved-period forecast inputs only after the release is frozen."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

def main():
    decision_path=ROOT/'release_decision.json'
    if not decision_path.exists():
        raise RuntimeError('Freeze the team release before preparing final-test inputs.')
    decision=json.loads(decision_path.read_text())
    if decision['status']!='frozen':raise RuntimeError('Release is not frozen.')
    if decision['bike_model']=='counts':
        print('Counts-only release. No forecast inputs are needed.')
        return
    import pandas as pd
    import download_forecast_temperature as temperature
    import download_forecast_jma as rain_wind
    out=ROOT/'reports/final_input_weather'
    out.mkdir(parents=True,exist_ok=True)
    temperature.OUT=rain_wind.OUT=out
    start,end='2023-09-24','2023-12-31'
    temperature.download(start,end,'gfs_final_test',allow_missing=decision.get('bike_missing_forecast_fallback')=='counts')
    rain_wind.download(start_date=start,end_date=end,run_name='jma_final_test')
    temp=pd.read_csv(out/'gfs_final_test_daily_temperature.csv',parse_dates=['date']).set_index('date')
    rain=pd.read_csv(out/'jma_final_test_daily.csv',parse_dates=['date']).set_index('date')
    fields=['forecast_temp_day2_mean','forecast_temp_day2_min','forecast_temp_day2_max',
            'forecast_jma_day2_precip_sum','forecast_jma_day2_wind_mean','forecast_jma_day2_wind_max']
    daily=temp.join(rain)[fields]
    assert daily.index.equals(pd.date_range(start,end))
    if daily.isna().any().any() and decision.get('bike_missing_forecast_fallback')!='counts':
        raise ValueError('Missing weather without a frozen fallback.')
    daily.to_csv(out/'forecasts.csv',float_format='%.6f',index_label='date')
    print(out/'forecasts.csv')
    print('Archived forecasts only. No trip targets were read or models changed.')

if __name__=='__main__':
    main()
