# main.py — FastAPI entrypoint for the 3D U-Net super-resolution service
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from prometheus_fastapi_instrumentator import Instrumentator

from app.model import super_resolve

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="MRI / µCT Super-Resolution API",
    description=(
        "Accepts a low-resolution .raw volume and returns a super-resolved .raw "
        "volume using a trained 3D U-Net (sr_pipeline_v04 architecture)."
    ),
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)


@app.get("/healthz", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready():
    """Returns 503 until the model is loaded."""
    try:
        from app.model import get_model
        get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "ready"}


@app.post("/super-resolve", tags=["inference"])
async def run_super_resolution(
    file: UploadFile = File(..., description="LR volume as raw bytes (.raw file)"),  # noqa: B008
    x: int     = Query(...,  description="LR volume width  (voxels)"),
    y: int     = Query(...,  description="LR volume height (voxels)"),
    z: int     = Query(...,  description="LR volume depth  (voxels)"),
    dtype: str = Query("float32", description="LR voxel dtype: float32 | uint16 | int16"),
    lr_spacing: float = Query(...,  description="LR voxel size in µm (e.g. 5.38)"),
    hr_spacing: float = Query(...,  description="HR voxel size in µm (e.g. 0.65)"),
    lr_norm_lo: float | None = Query(None, description="LR normalisation lower bound (optional)"),
    lr_norm_hi: float | None = Query(None, description="LR normalisation upper bound (optional)"),
    out_dtype: str = Query("float32", description="Output dtype: float32 | uint16"),
    overlap:    int = Query(16,  description="Tile overlap in HR voxels"),
    batch_size: int = Query(4,   description="Tiles per GPU batch"),
):
    """
    Upload a `.raw` LR volume and receive a super-resolved `.raw` HR volume.

    **Filename convention for the response**: `<original_name>_SR_float32_<x>x<y>x<z>.raw`

    **Example** (curl):
    ```bash
    curl -X POST \\
      "http://localhost:8000/super-resolve?x=200&y=200&z=100&lr_spacing=5.38&hr_spacing=0.65" \\
      -F "file=@scan_float32_200x200x100.raw" \\
      --output scan_SR.raw
    ```
    """
    allowed_dtypes = {'float32', 'float64', 'uint16', 'int16', 'uint8'}
    if dtype not in allowed_dtypes:
        raise HTTPException(400, detail=f"dtype must be one of {allowed_dtypes}")
    if out_dtype not in {'float32', 'uint16'}:
        raise HTTPException(400, detail="out_dtype must be float32 or uint16")
    if lr_spacing <= 0 or hr_spacing <= 0:
        raise HTTPException(400, detail="spacing values must be positive")
    if lr_spacing <= hr_spacing:
        raise HTTPException(400, detail="lr_spacing must be larger than hr_spacing (upsampling)")

    lr_norm = (lr_norm_lo, lr_norm_hi) if (lr_norm_lo is not None and lr_norm_hi is not None) else None

    lr_bytes = await file.read()
    expected = x * y * z * __import__('numpy').dtype(dtype).itemsize
    if len(lr_bytes) != expected:
        raise HTTPException(
            400,
            detail=(
                f"File size {len(lr_bytes)} bytes does not match "
                f"{x}×{y}×{z} {dtype} ({expected} bytes expected)."
            ),
        )

    try:
        hr_bytes, hr_dims_xyz, hr_dtype = super_resolve(
            lr_bytes=lr_bytes,
            dims_xyz=(x, y, z),
            dtype=dtype,
            lr_spacing=lr_spacing,
            hr_spacing=hr_spacing,
            lr_norm=lr_norm,
            out_dtype=out_dtype,
            overlap=overlap,
            batch_size=batch_size,
        )
    except FileNotFoundError as e:
        raise HTTPException(503, detail=str(e))

    hx, hy, hz = hr_dims_xyz
    base = os.path.splitext(file.filename or "volume")[0]
    out_name = f"{base}_SR_{hr_dtype}_{hx}x{hy}x{hz}.raw"

    return Response(
        content=hr_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
