FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN <<EOF
apt update
apt install -y python3-full python3-venv python3-pip
apt clean && rm -rf /var/lib/apt/lists/*

pip install -U uv --break-system-packages
pip cache purge
EOF

WORKDIR /server

COPY pyproject.toml uv.lock ./

RUN <<EOF
uv sync --no-dev
uv pip list
uv cache clean
EOF

ENV PATH="/server/.venv/bin:$PATH"

COPY utils/ utils/

COPY async_server.py pipelines.py ./

ENTRYPOINT []
