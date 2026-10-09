FROM python:3.12.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 UV_PYTHON_INSTALL_DIR=/opt/python UV_CACHE_DIR=/tmp/uv-cache
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-citibike.txt requirements-mushroom-serving.txt ./
RUN python -m pip install uv && uv python install 3.11.13 --no-bin \
    && uv venv .venv-pycaret --python 3.11.13 \
    && uv pip install --python .venv-pycaret/bin/python -r requirements-mushroom-serving.txt \
    && python -m venv .venv \
    && .venv/bin/python -m pip install -r requirements-citibike.txt
COPY app/ app/
COPY src/ src/
COPY scripts/run_local_app.py scripts/run_local_app.py
COPY citibike/development_daily.csv citibike/development_daily_version.json citibike/archive_manifest.json citibike/evaluation_plan.json citibike/
COPY models/mushroom_candidate.joblib models/citibike_candidate.joblib models/
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD .venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"
CMD [".venv/bin/python", "scripts/run_local_app.py", "--host", "0.0.0.0", "--port", "8080"]
