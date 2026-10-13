FROM python:3.12.10-slim
WORKDIR /project
COPY requirements-review.txt requirements-neural-review.txt ./
RUN pip install --no-cache-dir -r requirements-review.txt \
    && pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY . .
EXPOSE 8780
CMD ["python", "app/review_server.py", "--host", "0.0.0.0", "--port", "8780"]
