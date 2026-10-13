# Run and release

## Local app

```powershell
.\.venv-review\Scripts\python.exe app/review_server.py
```

Open http://127.0.0.1:8780. The page sends inputs to the Python API. Saved pipelines return predictions. Missing weather inputs produce an error.

Use the environments listed in [the notebook map](NOTEBOOKS.md). Run `scripts/verify_review.py` with the review interpreter to check saved predictions, API paths and invalid inputs.

## AWS evidence - pending

Run [mushroom notebook 10](../mushrooms/10_aws_extra_trees.ipynb) and [bike CPU notebook 16](../citibike/16_aws_cpu.ipynb) on an allowed SageMaker notebook instance. [Bike GPU notebook 17](../citibike/17_aws_gru_optional.ipynb) is optional. Enter the actual operator, save executed notebooks and export each evidence ZIP. Uploading a local saved model is not cloud training.

Keep cloud results separate from the local reports. If adopting an AWS model, update its artifact and manifest together. A refitted mushroom model needs its own calibrated cutoff. Confirm the AWS permissions and minimum run requirement before starting.

## Freeze and final tests - pending

Agree release models, required inputs and fit boundaries. Run `scripts/freeze_release.py` to record their hashes before opening reserved test targets.

Bring the exact teacher mushroom CSV and audited annual bike table. For a weather model, use `scripts/download_final_test_forecasts.py` after freezing; do not substitute observed weather. Then run `scripts/evaluate_final_tests.py` once. Its marker blocks repeat scoring.

Example for a counts-only release, after the team makes that choice:

```powershell
.\.venv-review\Scripts\python.exe scripts/freeze_release.py --bike-model counts --operator "YOUR NAME" --reason "Team release decision"
.\.venv-review\Scripts\python.exe scripts/evaluate_final_tests.py --mushroom-csv "PATH TO TEACHER CSV" --bike-daily "PATH TO ANNUAL DAILY CSV" --acknowledge-final-test
```

Use `weather_cpu` or `weather_gru` for a weather release, and add `--forecasts "PATH TO FORECAST DAILY CSV"` when scoring. Keep predictions and `reports/final_test_results.json`, even if results are worse than development. A failed run needs review before any retry.

## Hosting

```bash
docker build -t cloudai .
docker run --rm -p 8780:8780 cloudai
```

The Docker build passed on GitHub. Choose a host, connect its public HTTPS address to port 8780 and test both prediction paths. Public hosting and restart behaviour remain unverified.

## Automatic updates

The GitHub workflow passed its CPU model rebuild, app checks and Docker build. The GPU-trained GRU artifact stays frozen. A new mushroom fit and its recalibrated threshold travel together in `model_manifest.json`.

The host must retrieve the rebuilt artifact for the pushed commit and restart the app with it. Configure its hook as repository secret `DEPLOY_WEBHOOK`. A hook alone does not install the artifact. Test an actual update before marking deployment done. CI is not AWS training evidence.

Save the public URL, exact release commit and restart instructions here. A teammate must reproduce the release and verify an actual update. Then save the final notebook outputs and submit the repository link in Canvas before 15 October, 23:59.
