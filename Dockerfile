FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /workspace
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project
ENV CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/workspace/.venv/bin/python \
    CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1
