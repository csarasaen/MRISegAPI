# MRI / µCT Super-Resolution Platform

![CI](https://github.com/csarasaen/MRISegAPI/actions/workflows/ci-cd.yml/badge.svg)

A **production-style cloud deployment** of a 3D U-Net super-resolution service:
a FastAPI inference endpoint wrapping 'sr pipeline', containerised with Docker,
deployed to Kubernetes on AWS EKS, provisioned via Terraform, and monitored with
Prometheus + Grafana.

The model upsamples low-resolution µCT/MRI `.raw` volumes to the HR grid
(~(LR_spacing/HR_spacing)³ × more voxels) using a residual 3D U-Net trained
with L1+SSIM loss.

## Architecture

```
LR .raw  ──▶  FastAPI /super-resolve  ──▶  3D U-Net (TF)  ──▶  HR .raw
                       │
                 Prometheus /metrics
                       │
               Grafana dashboard
```

Deployed on AWS EKS (GPU Spot nodes), provisioned with Terraform, and continuously
delivered via GitHub Actions and Azure Pipelines.

## What's in this repo

| Folder               | What it does                                          |
|----------------------|-------------------------------------------------------|
| `app/`               | FastAPI SR service, 3D U-Net wrapper, tests           |
| `app/srtools/`       | Adapted inference-only copy of 'sr pipeline'	       |	
| `terraform/aws/`     | EKS cluster + VPC (Terraform)                         |
| `kubernetes/`        | Deployment, Service, HorizontalPodAutoscaler          |
| `.github/workflows/` | CI/CD: lint → test → build → push → deploy            |
| `azure-pipelines/`   | Azure DevOps pipeline (mirrors GitHub Actions)        |
| `monitoring/`        | Prometheus + Grafana docker-compose stack             |
| `scripts/`           | Health-check automation                               |

## Prerequisites

- Docker Desktop running
- Trained model files (from `UNet3D_SR_training.py`):
  ```
  models/
    Unet_SR_UNet3D.json
    Unet_SR_UNet3D_final.weights.h5
  ```

## Run locally with Docker

```bash
# 1. Build the image (from the repo root)
docker build -t mri-sr:latest ./app

# 2. Run, mounting your model weights
docker run -p 8000:8000 \
  -v "C:/path/to/your/models:/models" \
  mri-sr:latest

# 3. Open the interactive API docs
#    http://localhost:8000/docs
```

## Call the API

```bash
curl -X POST \
  "http://localhost:8000/super-resolve?x=200&y=200&z=100&lr_spacing=5.38&hr_spacing=0.65" \
  -F "file=@LR_scan_float32_200x200x100.raw" \
  --output SR_output.raw
```

Or open `http://localhost:8000/docs` and use the Swagger UI.

## Run the monitoring stack

```bash
cd monitoring
docker compose up --build

# API:        http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000
```

## Run the tests (no model weights needed)

```bash
pip install -r app/requirements.txt pytest httpx
pytest app/tests/ -v
```

## Key design decisions

- **Spot instances** for GPU node group — ~70 % cost reduction vs. on-demand
- **Single NAT gateway** in non-prod — saves ~€30/month per environment
- **HPA** scales 2 → 20 pods on CPU/memory — handles burst scan loads
- **Non-root container** — security baseline for medical-data workloads
- **Model weights NOT baked into the image** — mounted as a volume; no credential
  leakage, weights can be updated without a rebuild
- **No training-time dependencies** — the API container is inference-only
