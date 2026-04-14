FROM python:3.11-slim

# /app is the project root inside the container.
# The serving package lives at /app/serving/ so that imports like
# `from serving.app.config import ...` resolve correctly — matching local dev.
WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the serving package
COPY . /app/serving/

EXPOSE 8000

CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
