# Reality Check simulation engine. The SAME image is used by participants for local practice
# (public cup tasks baked in) and by the scoring runner (hidden task packs mounted at /packs).
#
#   docker build -t rc-engine:dev engine/
#   docker run --rm -p 127.0.0.1:8090:8090 rc-engine:dev                      # practice, CPU or GPU
#   docker run --rm -p 127.0.0.1:8090:8090 --gpus all \
#       -e NVIDIA_DRIVER_CAPABILITIES=all rc-engine:dev                       # GPU (EGL) rendering
#   docker run ... -v /srv/rc/packs:/packs:ro rc-engine:dev --task-packs /packs   # scoring (hidden tasks)
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglx-mesa0 libgl1-mesa-dri libosmesa6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 RC_ENGINE_HOST=0.0.0.0 RC_ENGINE_PORT=8090 \
    RC_EMBODIMENT=ur12e_robotiq_2f140
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY rc_engine ./rc_engine
COPY docker-entrypoint.sh /usr/local/bin/rc-engine-entrypoint
RUN chmod +x /usr/local/bin/rc-engine-entrypoint

ARG RC_ENGINE_REVISION=dev
LABEL org.opencontainers.image.title="rc-engine" org.opencontainers.image.revision=$RC_ENGINE_REVISION
EXPOSE 8090
ENTRYPOINT ["rc-engine-entrypoint"]
