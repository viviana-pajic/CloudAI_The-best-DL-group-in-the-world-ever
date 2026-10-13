# CloudAI

The best DL group in the world ever.

Viviana Pajic, Tomislav Novosel, Muneeb Shakoor.

Deadline: 15 October 2026. Oral: 23 October.

## Run the app

Python 3.12.10. On Windows, run `scripts/setup_review.ps1` once, then:

```powershell
.\.venv-review\Scripts\python.exe app/review_server.py
```

Open http://127.0.0.1:8780.

## What we found

- Mushrooms: Extra Trees caught 300/303 poison labels, but flagged 474/497 edible labels. It is not a useful zero-miss classifier.
- Bike rides: the weather GRU blend has the lowest recorded MAE, 13,263 rides/day. The CPU weather blend has 13,494. Counts-only XGBoost has 14,761. The apparent improvement is uncertain.
- These are development results. The final tests are untouched.

## Read in order

Follow [the notebook map](docs/NOTEBOOKS.md). Each notebook covers one step.

[Simple explanation](docs/TEAM_GUIDE.md) · [AWS steps](docs/AWS.md) · [Tasks](docs/TASKS.md) · [Run and host](docs/RELEASE.md) · [AI use](docs/AI_USE.md)

AWS, public hosting, automatic deployment and final tests remain unfinished.
