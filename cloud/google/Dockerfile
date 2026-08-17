# Build this image only after the production preflight has passed.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace/raiq
COPY pyproject.toml requirements.txt README.md ./
COPY raiq ./raiq
COPY configs ./configs
COPY data/manifests ./data/manifests
COPY scripts ./scripts

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# A managed training job should override command arguments with its approved run config.
ENTRYPOINT ["python", "-m", "raiq.training.train"]
