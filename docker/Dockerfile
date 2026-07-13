# FREUID Challenge 2026 - team junesdata verification image.
# Organizers run: docker run --rm --network none --gpus all \
#   -v DATA:/data:ro -v OUT:/submissions freuid-solution

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FREUID_DATA_DIR=/data \
    FREUID_OUTPUT_DIR=/submissions \
    FREUID_SUBMISSION_PATH=/submissions/submission.csv \
    FREUID_WEIGHTS_DIR=/app/weights

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prepare_submission.py /app/
COPY weights/ /app/weights/

RUN useradd --create-home --uid 1000 runner
USER runner

ENTRYPOINT ["python", "/app/prepare_submission.py"]
CMD []
