FROM python:3.11-slim

# System dependencies for headless MuJoCo (EGL) + useful debugging tools
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        libegl1 libgles2 libosmesa6 libgl1-mesa-dri \
        git curl htop strace vim procps net-tools && \
    rm -rf /var/lib/apt/lists/*

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv

WORKDIR /app

# Layer-cached dependency install: copy lockfile first, install deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy source and install project
COPY . .
RUN uv sync --frozen

ENV MUJOCO_GL=egl
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "python", "main.py", "train"]
