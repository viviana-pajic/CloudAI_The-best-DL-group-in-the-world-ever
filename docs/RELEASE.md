# Run and host

## Local app

```powershell
.\.venv-review\Scripts\python.exe app/review_server.py
```

Open http://127.0.0.1:8780. The page sends inputs to the Python API. Saved pipelines return predictions. Missing weather inputs produce an error.

## Hosting

```bash
docker build -t cloudai .
docker run --rm -p 8780:8780 cloudai
```

Choose a host, connect its public HTTPS address to port 8780 and test both prediction paths. Docker and public hosting are still unverified.

## Automatic updates

The prepared GitHub workflow rebuilds CPU models and checks the app. The GPU-trained GRU artifact stays frozen. A new mushroom fit and its recalibrated threshold travel together in `model_manifest.json`.

The host must retrieve the rebuilt artifact for the pushed commit and restart the app with it. Configure its hook as repository secret `DEPLOY_WEBHOOK`. A hook alone does not install the artifact. Test an actual update before marking deployment done. CI is not AWS training evidence.

After the handoff, [freeze the release and run final tests](FINAL_TEST_PLAN.md). Save the public URL and restart instructions here once they exist.
