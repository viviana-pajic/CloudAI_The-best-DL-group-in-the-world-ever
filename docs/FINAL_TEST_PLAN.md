# Final test plan

Status: prepared, not run. No release decision has been frozen in this review folder.

1. Review local and actual AWS results. Agree the release models, input requirements, fit boundaries and mushroom threshold procedure.
2. If replacing a model with an AWS fit, update the artifact and model manifest together. Use the AWS model's own calibrated threshold.
3. Record the team's choice using `scripts/freeze_release.py`. It saves file hashes and refuses to overwrite an existing choice.
4. Bring the exact teacher mushroom CSV and the audited full 2023 daily Citi Bike CSV. The original acquisition notebooks recreate them. Keep raw trip files out of GitHub.
5. For a weather model, run `scripts/download_final_test_forecasts.py` after freezing. It downloads the same archived forecast fields and aggregation for 24 September-31 December, including the CPU model's seven-day history. It reads no ride targets and writes `reports/final_input_weather/forecasts.csv`. This final-period download is prepared, not run. Do not substitute observed weather.
6. Run `scripts/evaluate_final_tests.py` once with `--acknowledge-final-test`. Its marker prevents repeat scoring.
7. Preserve the predictions and `reports/final_test_results.json`. Include them in the final comparison notebooks. Report the results even if they are worse than development.

Example after the team has chosen counts-only XGBoost:

```powershell
.\.venv-review\Scripts\python.exe scripts/freeze_release.py --bike-model counts --operator "YOUR NAME" --reason "Team release decision"
.\.venv-review\Scripts\python.exe scripts/evaluate_final_tests.py --mushroom-csv "PATH TO TEACHER CSV" --bike-daily "PATH TO ANNUAL DAILY CSV" --acknowledge-final-test
```

For a weather choice, add `--forecasts "PATH TO FORECAST DAILY CSV"`. Choose `weather_cpu` or `weather_gru` when freezing.

The script is prepared and syntax-checked. It has deliberately not been run against the reserved data. Keep the exclusive marker after a run. If a run fails, review the failure and what was exposed before deciding how to recover.
