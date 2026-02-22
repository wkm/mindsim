FROM python:3.11-slim

# System dependencies for headless MuJoCo (EGL)
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        libegl1 libgles2 libosmesa6 libgl1-mesa-dri && \
    rm -rf /var/lib/apt/lists/*

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv

WORKDIR /app

# Copy source and lockfile (deps installed at startup)
COPY . .

ENV MUJOCO_GL=egl
ENV PYTHONUNBUFFERED=1

# Install deps and run training at startup
ENTRYPOINT ["sh", "-c", "uv sync --frozen && exec uv run python main.py train \"$@\"", "--"]
