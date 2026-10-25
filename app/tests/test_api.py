# test_api.py — API smoke tests (no model weights required)
#
# These tests mock super_resolve so the suite runs in CI without a GPU or model files.
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_raw(x=10, y=10, z=5, dtype='float32') -> bytes:
    arr = np.random.rand(z, y, x).astype(dtype)
    return arr.tobytes()


# ── ops endpoints ──────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_without_model_returns_503():
    """Without model weights mounted, /ready should return 503."""
    r = client.get("/ready")
    assert r.status_code == 503


# ── inference endpoint ─────────────────────────────────────────────────────────

def _mock_super_resolve(lr_bytes, dims_xyz, dtype, lr_spacing, hr_spacing, **kwargs):
    x, y, z = dims_xyz
    scale = lr_spacing / hr_spacing
    hx, hy, hz = int(x * scale), int(y * scale), int(z * scale)
    hr = np.zeros(hz * hy * hx, dtype='float32').tobytes()
    return hr, (hx, hy, hz), 'float32'


def test_super_resolve_returns_raw():
    raw = _make_raw(x=10, y=10, z=5)
    with patch("app.main.super_resolve", side_effect=_mock_super_resolve):
        r = client.post(
            "/super-resolve",
            params={"x": 10, "y": 10, "z": 5, "lr_spacing": 5.38, "hr_spacing": 0.65},
            files={"file": ("scan_float32_10x10x5.raw", raw, "application/octet-stream")},
        )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert "SR" in r.headers["content-disposition"]


def test_wrong_dtype_rejected():
    r = client.post(
        "/super-resolve",
        params={"x": 10, "y": 10, "z": 5, "lr_spacing": 5.38, "hr_spacing": 0.65,
                "dtype": "complex128"},
        files={"file": ("scan.raw", b"x" * 100, "application/octet-stream")},
    )
    assert r.status_code == 400


def test_wrong_size_rejected():
    r = client.post(
        "/super-resolve",
        params={"x": 10, "y": 10, "z": 5, "lr_spacing": 5.38, "hr_spacing": 0.65},
        files={"file": ("scan.raw", b"x" * 7, "application/octet-stream")},
    )
    assert r.status_code == 400


def test_lr_larger_than_hr_required():
    raw = _make_raw(x=10, y=10, z=5)
    r = client.post(
        "/super-resolve",
        params={"x": 10, "y": 10, "z": 5, "lr_spacing": 0.5, "hr_spacing": 5.0},
        files={"file": ("scan_float32_10x10x5.raw", raw, "application/octet-stream")},
    )
    assert r.status_code == 400
