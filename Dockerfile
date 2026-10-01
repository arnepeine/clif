FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CLIF_DATA_DIR=/data

RUN groupadd --gid 10001 clif && useradd --uid 10001 --gid clif --create-home clif \
    && mkdir -p /data && chown clif:clif /data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY clif ./clif
# Extras are opt-in at build time, e.g. --build-arg CLIF_EXTRAS="laya,faiss"
ARG CLIF_EXTRAS=""
RUN pip install ".${CLIF_EXTRAS:+[$CLIF_EXTRAS]}" && pip check

USER clif
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "clif.api:app", "--host", "0.0.0.0", "--port", "8000"]
