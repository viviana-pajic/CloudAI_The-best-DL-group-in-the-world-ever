FROM python:3.12.10-slim
WORKDIR /project
COPY requirements-review-linux.lock.txt ./
RUN pip install --no-cache-dir -r requirements-review-linux.lock.txt \
    && pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY . .
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN if [ -f release_decision.json ]; then \
      python scripts/assemble_hosting.py && python scripts/verify_review.py --mode frozen; \
    else \
      python scripts/rebuild_review_models.py --output-root /tmp/cloudai-build \
      && python scripts/assemble_hosting.py --candidate-root /tmp/cloudai-build \
      && python scripts/verify_review.py --mode rebuilt; \
    fi
EXPOSE 10000
CMD ["python","app/review_server.py","--host","0.0.0.0"]
