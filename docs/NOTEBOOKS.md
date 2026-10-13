# Notebook order

Read saved outputs in this order. The original class comparisons and later research use different splits; compare scores within their own protocol. AWS notebooks 10 and 16 have saved cloud outputs from 13 October.

## Mushrooms

| Notebook | Step |
|---|---|
| [00](../mushrooms/00_data_audit.ipynb) | Replacement dataset and EDA |
| [01](../mushrooms/01_data_preparation.ipynb) | Class preprocessing |
| [02](../mushrooms/02_baseline_models.ipynb) | Baselines |
| [03](../mushrooms/03_automl.ipynb) | PyCaret comparison |
| [04](../mushrooms/04_model_comparison.ipynb) | Class-model errors and thresholds |
| [05](../mushrooms/05_tune_logistic_regression.ipynb) | Logistic-regression tuning |
| [06](../mushrooms/06_tune_random_forest.ipynb) | Random-forest tuning |
| [07](../mushrooms/07_selected_preparation.ipynb) | Selected preprocessing |
| [08](../mushrooms/08_extra_trees.ipynb) | Extra Trees training and calibration |
| [09](../mushrooms/09_model_comparison.ipynb) | Frozen research comparison |
| [10](../mushrooms/10_aws_extra_trees.ipynb) | AWS Extra Trees training, tuning and cutoff - executed |
| [11](../mushrooms/11_final_test.ipynb) | Frozen release, final results and alarm tradeoff |

## Citi Bike

| Notebook | Step |
|---|---|
| [00](../citibike/00_data_acquisition.ipynb) | Trip acquisition |
| [01](../citibike/01_data_audit.ipynb) | Daily counts and EDA |
| [02](../citibike/02_weekly_hypothesis.ipynb) | Weekly-pattern hypothesis |
| [03](../citibike/03_data_preparation.ipynb) | Class forecast features |
| [04](../citibike/04_baselines.ipynb) | Simple baselines |
| [05](../citibike/05_automl.ipynb) | FLAML comparison |
| [06](../citibike/06_ridge.ipynb) | Ridge tuning |
| [07](../citibike/07_xgboost.ipynb) | XGBoost tuning |
| [08](../citibike/08_random_forest.ipynb) | Random-forest tuning |
| [09](../citibike/09_error_analysis.ipynb) | Difficult dates and errors |
| [10](../citibike/10_model_comparison.ipynb) | Class-model comparison |
| [11](../citibike/11_forecast_data.ipynb) | Added history and forecast sources |
| [12](../citibike/12_selected_preparation.ipynb) | Selected forecast features |
| [13](../citibike/13_weather_svr_hist.ipynb) | CPU weather blend |
| [14](../citibike/14_weather_gru_catboost.ipynb) | GRU weather blend |
| [15](../citibike/15_model_comparison.ipynb) | Frozen research comparison |
| [16](../citibike/16_aws_cpu.ipynb) | AWS CPU blend training and tuning - executed |
| [17](../citibike/17_aws_gru_optional.ipynb) | Optional AWS GPU training - not needed for release |
| [18](../citibike/18_final_test.ipynb) | Frozen release, final forecasts, fallback and errors |

The app uses bike research candidates trained through June. Citi Bike 10 retains the earlier September-fitted class pilot.

## Environments

| Work | Python | Requirements |
|---|---|---|
| Mushrooms 00-06 | 3.11.13 | `requirements-pycaret.lock.txt` |
| Citi Bike 00-10 | 3.12.10 | `requirements-citibike.lock.txt` |
| Later notebooks and app | 3.12.10 | `requirements-review.txt` |

Run `scripts/setup_review.ps1` for the current Windows app environment. PyCaret keeps its separate environment. Saved GRU inference works on CPU; our GRU training code requires CUDA.
