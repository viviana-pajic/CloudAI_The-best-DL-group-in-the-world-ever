# Run and release

## Frozen models and final results

The release uses the original Extra Trees fit with cutoff 0.0398015873015873. Its default 0.5 rule remains a comparison. The bike release uses 75% GRU / 25% CatBoost, with counts-only XGBoost when weather forecasts are missing. Other bike choices remain development references.

These fits were selected from existing development results. The first freeze failed weather preflight: GFS had 53 unavailable required hours on 29-31 December. Its decision is preserved in `reports/release_preflight_decision.json`. Before reading test targets, we recorded the fallback and froze again in `release_decision.json`. No model choice changed after scoring.

Final tests completed once on 13 October: 1,000 mushroom rows and 92 October-December bike dates. The lower rule caught 373/378 poison labels and falsely flagged 598/622 edible labels. The bike release MAE was 15,005; last week's count had MAE 24,255. The GRU handled 89 dates; XGBoost handled 29-31 December. Notebooks 11 and 18 read saved results, not new test runs.

`reports/final_test_started.json` prevents repeat scoring. Keep it and all predictions, even if results disappoint. Do not rerun the evaluator or tune against these results. Text fingerprints normalize Windows/Linux line endings; binary model hashes remain byte-exact.

## Local app and checks

Python 3.12.10. Run `scripts/setup_review.ps1` once on Windows, then:

```powershell
.\.venv-review\Scripts\python.exe app/review_server.py
.\.venv-review\Scripts\python.exe scripts/verify_review.py --mode frozen
```

Open http://127.0.0.1:8780. The page sends inputs to the Python API. Saved pipelines return predictions. An empty forecast list selects the GRU's counts-only fallback. Partial or invalid forecasts produce an error.

Use the environments in [NOTEBOOKS.md](NOTEBOOKS.md). Startup verifies frozen prediction source and model hashes. The requested random-example interface update is recorded in `release_ui_update.json`; the original decision and final results remain unchanged. The app reads final scores from `reports/final_test_results.json`; it never scores test targets on a browser request. The manifest's `final_tests_scored: false` describes the original artifact-copy stage. Current test status is in the final report and `/health`.

## AWS evidence

Tomislav ran notebooks 10 and 16 on the university `MyNotebook` in `us-east-1` (`ml.m4.xlarge`) on 13 October. Codex assisted. Executed notebooks and [saved runs](../reports/aws/) contain actual outputs, tuning records and trained models. Evidence ZIPs were downloaded. The sandbox denied the normal Stop action; the lab remains open at Tomislav's request. End Lab deletes temporary resources.

AWS Extra Trees selected 600 trees; training-fold average precision was 0.7776. Its calibration caught 242/242 poison labels and flagged 380/398 edible labels. Calibration chose the cutoff; it is not an independent test.

The AWS CPU bike blend selected 200 boosting iterations; training-fold MAE was 9,710.58. These scores use different rows from the local development comparisons. AWS refits stay separate; their evidence does not establish better final performance. Optional GPU notebook 17 is not required to serve the saved GRU on CPU.

## Hosting and automatic updates

[Public app](https://cloudai-team.onrender.com). Render service `cloudai-team`: Free, Frankfurt, no payment method. It sleeps after 15 idle minutes and usually needs about a minute to wake. Open it before the demo. [Render free limits](https://render.com/docs/free).

The server uses `PORT` (10000 on Render); locally it defaults to 8780. Manual restart: Render dashboard > cloudai-team > Manual Deploy > Deploy latest commit. Hosting does not depend on BESTIJA or the AWS notebook remaining on.

```bash
docker build -t cloudai .
docker run --rm -p 8780:8780 cloudai
```

A code push runs a training-only CPU candidate build, release checks and a Docker build. Candidate tuning records remain separate build artifacts. While `release_decision.json` exists, Docker packages the exact frozen files from `models/`; it does not replace them with a fresh fit. This protects the tested release. Promoting a future candidate needs a new evaluation plan, not reuse of this final set.

After checks pass on main, GitHub requests deployment of that exact commit using the private `DEPLOY_WEBHOOK` secret. Render's native auto-deploy is off. No secret is committed. The earlier model rebuild and automatic update passed: [run 38448457190](https://github.com/viviana-pajic/CloudAI_The-best-DL-group-in-the-world-ever/actions/runs/38448457190), [evidence](../reports/hosting_update_verification.json).

`/health` shows the serving commit, all four model hashes, frozen decision hash and test completion. `/api/release` returns saved final scores. Public verification compares hashes, saved scores and training-side example predictions with the repository. It also checks invalid inputs and the missing-weather fallback. It does not rerun final tests.

```bash
python scripts/verify_hosted.py --url https://cloudai-team.onrender.com --expected-commit FULL_SERVING_COMMIT --output reports/hosting_final_verification.json
```

Later documentation-only commits may be newer than the serving code commit. The final verification report records the checked version.

Verified 13 October: serving commit `d8d44dede9373151f828a626b2fc86fd95740e0a`. [GitHub run 38452685016](https://github.com/viviana-pajic/CloudAI_The-best-DL-group-in-the-world-ever/actions/runs/38452685016) passed its fresh Linux environment, CPU candidate training, frozen app checks and Docker build, then triggered Render deployment `dep-db785oavcj2c73cjo45g`.

[Public verification](../reports/hosting_final_verification.json) passed: all four model hashes match GitHub, decision hashes and saved final scores match, all prediction choices match local example outputs, missing-weather fallback works and invalid inputs are rejected. Both browser buttons and the final score section worked. [CI evidence](../reports/release_ci_verification.json) and [clean checkout checks](../reports/clean_checkout_verification.json) are saved. The clean Windows checkout used an existing pinned environment; CI installed its Linux environment from scratch. Teammate explanation and reproduction remain separate human checks.

## Before submission

- Each teammate reviews both final notebooks, AWS outputs and the demo.
- Rehearse explaining the inputs, splits, missing values, thresholds and fallback.
- Submit the repository link in Canvas by 15 October 2026, 23:59. Verify lecturer access and save the receipt.
- Keep free hosting available for assessment and check it before the 23 October oral.
Random-example update: both buttons select real training-side inputs and exclude the previous example. Fixed example endpoints remain available for repeatable checks. `scripts/verify_random_examples.py` checks pool validity, changing examples and prediction parity without fitting or scoring reserved targets.
