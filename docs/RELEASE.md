# Run and release

## Local app

```powershell
.\.venv-review\Scripts\python.exe app/review_server.py
```

Open http://127.0.0.1:8780. The page sends inputs to the Python API. Saved pipelines return predictions. Missing weather inputs produce an error.

Use the environments listed in [the notebook map](NOTEBOOKS.md). Run `scripts/verify_review.py` with the review interpreter to check saved predictions, API paths and invalid inputs.

## AWS evidence - completed 13 October

Tomislav ran notebooks 10 and 16 on the university `MyNotebook` in `us-east-1` (`ml.m4.xlarge`). Codex assisted. Each notebook contains actual outputs. [Saved runs](../reports/aws/) include tuning results, redacted provenance and trained models. Evidence ZIPs were downloaded before stopping compute.

Extra Trees selected 600 trees. Mean fold average precision: 0.7776. Its calibrated cutoff is 0.0398: 242/242 poisonous labels caught, 380/398 edible labels flagged. These rows chose the cutoff; this is not an independent test.

The CPU bike blend selected 200 histogram-boosting iterations. Mean chronological fold MAE: 9,710.58 rides. This is training-period CV, not a new summer result. No reserved final test was opened. The app's original local candidates and their historical comparisons remain separate from these AWS runs.

To repeat: run [mushroom notebook 10](../mushrooms/10_aws_extra_trees.ipynb) and [bike CPU notebook 16](../citibike/16_aws_cpu.ipynb) with the review kernel on an allowed SageMaker notebook instance. Enter the actual operator, save outputs and export each evidence ZIP. [Bike GPU notebook 17](../citibike/17_aws_gru_optional.ipynb) is optional. Uploading a local saved model is not cloud training.

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

`render.yaml` selects a free Docker service in Frankfurt. Connect the repository on Render and keep the free plan with no payment method. The server uses the host's `PORT` value (10000 on Render); locally it defaults to 8780. Public hosting and restart behaviour remain unverified until its first deploy passes.

The free service sleeps after 15 idle minutes and usually takes about a minute to wake. Open it before the demonstration. Linux checks loaded all model paths at about 437 MB; the free limit is 512 MB. Check the deployed service too. [Render free limits](https://render.com/docs/free).

## Automatic updates

The GitHub workflow passed its CPU model rebuild, app checks and Docker build. The GPU-trained GRU artifact stays frozen. A new mushroom fit and its recalibrated threshold travel together in `model_manifest.json`.

On Render, use automatic deploys after CI passes. The Docker build repeats the CPU training-only search and packages the resulting models and recalibrated cutoff. `reports/hosting_build.json` records the build. The GRU artifact stays frozen. These are hosting builds, not additional AWS runs.

`/health` shows the deployed commit, model hashes and cutoff. Check those after a pushed change and test both prediction paths before marking updates done. Render's native integration needs no `DEPLOY_WEBHOOK`. For another host, that optional workflow hook must actually install the built models; a hook alone does not do so.

Save the public URL, exact release commit and restart instructions here. A teammate must reproduce the release and verify an actual update. Then save the final notebook outputs and submit the repository link in Canvas before 15 October, 23:59.
