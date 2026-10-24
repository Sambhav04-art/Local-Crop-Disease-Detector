FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY cropdx/__init__.py cropdx/constants.py cropdx/preprocess.py cropdx/labels.py cropdx/
COPY api/ api/
COPY artifacts/ artifacts/
ENV MODEL_DIR=/app/artifacts
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
