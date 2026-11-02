# -----------------------------------------------------------------------------
#  AdventureWorks Analysis Dockerfile – Python 3.11 (slim) Base
#  Contribution margin, churn & markup simulation pipeline + JupyterLab
# -----------------------------------------------------------------------------

FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -----------------------------------------------------------------------------
#  System Packages (libgomp1 is required by LightGBM)
# -----------------------------------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
#  Python Requirements
# -----------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /opt/adventureworks
EXPOSE 8888
