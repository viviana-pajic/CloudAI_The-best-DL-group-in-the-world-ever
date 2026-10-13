# Data and evaluation

## Mushrooms

Use the teacher's replacement dataset: 5,000 rows and 13 columns. The source and fingerprint are recorded in [SOURCE_DATASET_VERSION.json](../SOURCE_DATASET_VERSION.json). The older proposed mushroom dataset is excluded.

Keep incomplete rows, zeros and large measurements; these are not proven errors. Fit filling and encoding inside training folds. Controlled comparisons supported removing two noise columns. Labels were not changed.

The class workflow uses 4,000 development rows and reserves 1,000 final-test rows. The research separates 2,559 search rows, 640 calibration rows and 800 checking rows. One row was quarantined to prevent an identical-input collision. Different protocols must not share one leaderboard.

## Citi Bike

[Published trip archives](https://s3.amazonaws.com/tripdata/index.html) and hashes are recorded in [source_config.json](../citibike/source_config.json) and [archive_manifest.json](../citibike/archive_manifest.json). Jersey City files are excluded.

Count by actual start date. January 2024 includes 410 rides that started in December 2023. Valid starts are retained even when unused station fields are empty. The audit found 35,107,120 rides across 365 dates, with no duplicate ride IDs or missing dates.

Use the previous 28 completed daily totals. Train through June, compare July-September, and reserve October-December. The GRU weather candidate also uses 2022 history. Quiet days are kept in the results.

## Weather and attribution

Use archived forecasts from Open-Meteo, NOAA GFS and JMA GSM, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Sources, hashes and raw forecast records are in [audit/weather](../audit/weather).

Our modifications aggregate in New York time, correct rain intervals and floor tiny negative rain forecasts. Exact historical publication times were not verified. Observed target-day weather is not substituted for forecasts. The app has no live count or weather feed.

All current results are development evidence. The reserved final tests remain unscored.

## Final test inputs

Final mushroom rows come from the original split manifest. Bike targets cover 1 October-31 December 2023. Archived GFS/JMA forecasts and request records are saved in `reports/final_input_weather/`. GFS has 53 unavailable required hours on 29-31 December; those three daily summaries stay missing. The release uses counts-only XGBoost on those dates. No observed weather, future ride totals or weather imputation was substituted. The 25-hour New York day on 5 November is retained. [Open-Meteo archive](https://open-meteo.com/en/docs/previous-runs-api).

## Random demo examples

The example buttons use 120 unique mushroom search-training rows and 153 bike dates from 29 January-30 June 2023. Bike inputs keep the matching earlier counts and archived forecasts. Labels and target-day ride counts are not shown. The previous example is excluded on the next click. Source hashes are in `examples/random_example_sources.json`. Reserved test targets were not read for this update.
