# Running and updating the shared app

## Local demo - task 17

The app has a home page and two prediction pages. Both run behind one address. Mushroom uses Python 3.11 and its existing reference forest at threshold 0.5; Citi Bike uses Python 3.12 and the new local candidate. Keeping their environments separate avoids incompatible package versions.

From the repository folder on Windows:

```powershell
py -3.11 -m venv .venv-pycaret
.\.venv-pycaret\Scripts\python.exe -m pip install -r requirements-pycaret.lock.txt
.\.venv-pycaret\Scripts\python.exe scripts/rebuild_mushroom_reference.py
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-citibike.lock.txt
.\.venv\Scripts\python.exe scripts/rebuild_citibike_candidate.py
.\.venv\Scripts\python.exe scripts/run_local_app.py
```

The verified versions are Python 3.11.13 and 3.12.10. On BESTIJA the environments and both model bundles have already been created, so only the last command is needed. Open **http://127.0.0.1:8780**. The two internal APIs use ports 8781 and 8782. The launcher stops its own child servers when you press Ctrl+C. To restart, run the last command again. Logs are under the ignored `.tools/app-logs/` folder.

Use `--port 8790` if the default ports are occupied. A remote laptop's `localhost` points to that laptop, not BESTIJA; use the desktop's browser through your existing remote session. This update has not exposed BESTIJA to the public internet.

The Citi Bike example supplies 28 made-up daily totals and a January 2024 forecast date. It demonstrates input formatting without predicting or revealing the reserved test targets. The mushroom page uses its existing training example.

## GitHub updates - part of task 30

The [Citi Bike workflow](../.github/workflows/citibike.yml) runs when relevant code, data snapshots or configuration change on GitHub. It installs the recorded Python environment, rebuilds the selected Citi Bike configuration on CPU, checks forecasting boundaries and saved-model/API agreement, and uploads a model bundle plus metadata named for the commit.

The workflow has read-only repository permissions. It does not push new commits, run the reserved test or deploy to a hosting account. A successful workflow is a reproducible model update, not a hosted application release. Complete the deployment step once a host, access method and release policy have been agreed.

The rebuild measures its own July-September validation result and stops if the candidate no longer beats the weekly baseline. The original GPU experiment scores remain in the tracked reports. CI artifacts retain the rebuilt model's actual package versions and metrics.

## Hosting preparation - task 29 remains open

A Dockerfile packages both model runtimes behind port 8080. Rebuild both bundles using the commands above before building; the bundles are ignored by Git but included in the container build context. Docker is not installed on BESTIJA, so this image has **not** been built or hosted here.

```text
docker build -t cloudai-demo .
docker run --rm -p 8080:8080 cloudai-demo
```

On a machine with Docker, verify both pages and `/health` locally before choosing a public host. The image contains historical models and no credentials. A public release still needs an agreed account, URL, cost limit, restart process and GitHub deployment connection. A teammate should verify the Linux container before it is treated as a release.

## AWS preparation - task 27 remains open

[Notebook 11](../citibike/11_aws_training.ipynb) and `scripts/train_citibike_aws.py` are prepared but **have not run on AWS**. They repeat the 192-configuration XGBoost search on a SageMaker notebook's CPU using the same training dates and folds. GPU rental is unnecessary for this small daily table. The helper requires actual SageMaker resource metadata before fitting and records a redacted resource ARN, source hash, environment, search results and validation predictions.

Once course access and spending limits are known, use the approved SageMaker notebook, clone this repository, create the documented Python 3.12 environment, and select it as the notebook kernel. Run notebook 11 and save its outputs and reports. Keep the console execution record for the assessment. AWS documents the [notebook metadata file](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-metadata.html).

Compare the actual AWS results with the local candidates before final selection. Do not replace the app model or score the final test automatically. Export the notebook/results, then stop the notebook's compute; [stopping a notebook instance ends its compute charges](https://docs.aws.amazon.com/cli/v1/reference/sagemaker/stop-notebook-instance.html), while provisioned storage or other resources may still cost money.
