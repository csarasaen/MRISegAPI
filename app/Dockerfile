# ── Base: TensorFlow CPU (swap tag to 2.16.1-gpu for GPU nodes) ───────────────
# CPU image is ~900 MB and runs anywhere; GPU image needs NVIDIA Container Toolkit.
FROM tensorflow/tensorflow:2.16.1 AS builder

WORKDIR /build
COPY requirements.txt .
# Install only the non-TF deps here (TF already in base image)
RUN pip install --no-cache-dir --prefix=/install \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    prometheus-fastapi-instrumentator==7.0.0 \
    "python-multipart==0.0.12" \
    scipy==1.13.1

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM tensorflow/tensorflow:2.16.1

WORKDIR /app

# Non-root user for security
RUN useradd --create-home appuser

COPY --from=builder /install /usr/local
COPY . .

# Model weights are NOT baked in — mount them at runtime via -v or K8s volume.
# Expected at /models/Unet_SR_UNet3D.json + /models/Unet_SR_UNet3D_final.weights.h5
VOLUME ["/models"]
ENV MODEL_PATH=/models

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
