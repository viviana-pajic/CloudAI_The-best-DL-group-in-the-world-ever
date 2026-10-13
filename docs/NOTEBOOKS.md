# Notebook order

Read saved outputs first. Original class notebooks and the later selected-model notebooks use different environments and splits. See [environments](ENVIRONMENTS.md).

## Mushrooms

| Notebook | One job | Main reason |
|---|---|---|
| 00 | Explore the replacement file | Missing inputs need rules |
| 01 | Prepare the class inputs | Keep filling inside folds |
| 02 | Compare baselines | Start with simple references |
| 03 | Run PyCaret comparison | Check other model families |
| 04 | Inspect class-model errors | AP alone does not describe the alarm |
| 05 | Tune logistic regression | Check whether tuning helps |
| 06 | Tune random forest | Improve the class candidate |
| 07 | Prepare the selected inputs | Add the useful logs and guarded ratios |
| 08 | Train Extra Trees | Reproduce the selected settings and recalibrate |
| 09 | Compare frozen research results | Explain misses and false alarms |
| 10 | Train and tune on AWS | Prepared; actual cloud run still needed |

## Citi Bike

| Notebook | One job | Main reason |
|---|---|---|
| 00 | Download and assemble trips | Count by actual start date |
| 01 | Explore daily counts | Check gaps and patterns |
| 02 | Test the weekly pattern | Support the prediction question |
| 03 | Prepare class features | Use only earlier counts |
| 04 | Compare simple guesses | Models must earn their place |
| 05 | Run FLAML comparison | Check other families |
| 06 | Tune Ridge | Try a simple numeric relationship |
| 07 | Tune XGBoost | Learn corrections with small trees |
| 08 | Tune random forest | Compare a different tree method |
| 09 | Inspect errors | Keep difficult days visible |
| 10 | Compare class models | Explain the earlier XGBoost choice |
| 11 | Check added history and forecasts | Weather adds a new input requirement |
| 12 | Prepare selected features | Preserve chronological boundaries |
| 13 | Train the CPU weather blend | Provide a practical AWS alternative |
| 14 | Explain the GRU weather blend | Lowest recorded MAE; optional CUDA reproduction |
| 15 | Compare frozen research results | Compare inputs, errors and uncertainty |
| 16 | Train and tune on AWS CPU | Prepared; actual cloud run still needed |
| 17 | Train and tune GRU on AWS | Optional, if the Sandbox permits a GPU |

The app uses saved research candidates trained through June. Notebook 10 retains the earlier September-fitted class pilot. Scores from different splits or fits must not be mixed.
