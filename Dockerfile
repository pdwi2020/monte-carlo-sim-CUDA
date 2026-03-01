# Monte Carlo Option Pricing - Docker Image
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Build Stage
# =============================================================================
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN pip3 install --no-cache-dir setuptools wheel pybind11 numpy

WORKDIR /app

COPY mc_pricer.py .
COPY setup.py .
COPY bates_kernel.cu .
COPY bates_kernel_extended.cu .
COPY bates_wrapper.cpp .
COPY bates_wrapper_extended.cpp .
COPY requirements.txt .

RUN python3 setup.py build_ext --inplace || echo "CUDA build skipped"

# =============================================================================
# Stage 2: Runtime Stage
# =============================================================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN useradd --create-home --shell /bin/bash mcuser
WORKDIR /app

COPY --from=builder /app/*.so /app/ 2>/dev/null || true
COPY --from=builder /app/mc_pricer.py /app/
COPY --from=builder /app/requirements.txt /app/

COPY risk_metrics.py /app/
COPY xva.py /app/
COPY calibration.py /app/
COPY cache.py /app/
COPY api/ /app/api/

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    pydantic>=2.0.0

RUN chown -R mcuser:mcuser /app
USER mcuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
