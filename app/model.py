# model.py — 3D U-Net super-resolution wrapper
#
# The model is loaded once at first use from MODEL_PATH (env var or default /models).
# Expected files inside MODEL_PATH:
#   Unet_SR_UNet3D.json              — architecture JSON (saved by UNet3D_SR_training.py)
#   Unet_SR_UNet3D_final.weights.h5  — trained weights
#
# To use a different name pair, set MODEL_JSON and MODEL_WEIGHTS env vars.
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_MODEL = None
_TF_READY = False
_MODEL_PATH    = os.environ.get('MODEL_PATH',    '/models')
_MODEL_JSON    = os.environ.get('MODEL_JSON',    'Unet_SR_UNet3D.json')
_MODEL_WEIGHTS = os.environ.get('MODEL_WEIGHTS', 'Unet_SR_UNet3D_final.weights.h5')


def _configure_tf():
    global _TF_READY
    if not _TF_READY:
        import tensorflow as tf
        for _gpu in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(_gpu, True)
            except RuntimeError:
                pass
        _TF_READY = True


def get_model():
    """Lazy-load the SR model (thread-safe via GIL for the assignment)."""
    global _MODEL
    if _MODEL is None:
        json_path    = os.path.join(_MODEL_PATH, _MODEL_JSON)
        weights_path = os.path.join(_MODEL_PATH, _MODEL_WEIGHTS)
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"Model architecture not found: {json_path}\n"
                f"Mount your model files to {_MODEL_PATH} and set MODEL_PATH if different."
            )
        from app.srtools import srTools
        _configure_tf()
        logger.info("Loading SR model from %s", _MODEL_PATH)
        _MODEL = srTools.loadModel((json_path, weights_path))
        logger.info("Model loaded. Input shape: %s", _MODEL.input_shape)
    return _MODEL


def super_resolve(
    lr_bytes: bytes,
    dims_xyz: tuple[int, int, int],
    dtype: str,
    lr_spacing: float,
    hr_spacing: float,
    lr_norm: tuple[float, float] | None = None,
    hr_norm: tuple[float, float] | None = None,
    out_dtype: str = 'float32',
    overlap: int = 16,
    batch_size: int = 4,
) -> tuple[bytes, tuple[int, int, int], str]:
    """Super-resolve a raw LR volume.

    Parameters
    ----------
    lr_bytes   : raw bytes of the LR volume (C-contiguous, dtype-encoded as given)
    dims_xyz   : (x, y, z) voxel counts of the LR volume
    dtype      : numpy dtype string of the LR data, e.g. 'float32', 'uint16'
    lr_spacing : LR voxel size in µm
    hr_spacing : HR voxel size in µm
    lr_norm    : (lo, hi) grey-value normalisation; None = estimate from data
    hr_norm    : (lo, hi) to rescale output; None = keep 0..1
    out_dtype  : dtype of the returned HR volume
    overlap    : tile overlap in HR voxels (16 is a good default for API use)
    batch_size : tiles per inference call

    Returns
    -------
    (hr_bytes, hr_dims_xyz, out_dtype)
        hr_bytes   : raw bytes of the super-resolved volume
        hr_dims_xyz: (x, y, z) of the HR output
        out_dtype  : dtype string of hr_bytes
    """
    from app.srtools import srData, srTools

    x, y, z = dims_xyz
    lr_arr = np.frombuffer(lr_bytes, dtype=np.dtype(dtype)).reshape(z, y, x)

    if lr_norm is None:
        lr_norm = srData.robustPercentiles(lr_arr)

    model = get_model()
    hr_arr = srTools.superResolveVolume(
        model, lr_arr,
        lrSpacing=lr_spacing,
        hrSpacing=hr_spacing,
        lrNorm=lr_norm,
        overlap=overlap,
        hrNorm=hr_norm,
        outDtype=out_dtype,
        batchSize=batch_size,
        nPrefetch=2,
        verbose=False,
    )

    hz, hy, hx = hr_arr.shape
    return hr_arr.tobytes(), (hx, hy, hz), out_dtype
