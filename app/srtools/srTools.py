"""TensorFlow/Keras side of the super-resolution pipeline — inference-only build."""
# ruff: noqa: N999
import itertools
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tensorflow import keras

from app.srtools import srData

def saveModel(model, saveName):
    """saveName: '.keras' file, or (json_path, weights_path) tuple."""
    if isinstance(saveName, tuple):
        with open(saveName[0], 'wt') as f:
            f.write(architectureJson(model))
        model.save_weights(saveName[1])
    else:
        model.save(saveName)


def architectureJson(model):
    cfg = json.loads(model.to_json())
    cfg.pop('compile_config', None)
    return json.dumps(cfg)


def loadModel(saveName):
    """Load a saved SR model.
    saveName: (json_path, weights_path) tuple, or a single .keras file path."""
    if isinstance(saveName, tuple):
        with open(saveName[0]) as f:
            cfg = json.load(f)
        cfg.pop('compile_config', None)
        model = keras.models.model_from_json(json.dumps(cfg))
        model.load_weights(saveName[1])
        return model
    return keras.models.load_model(saveName, compile=False)


# ─── losses ───────────────────────────────────────────────────────────────────

def _gaussWindow(size=11, sigma=1.5):
    g = np.exp(-0.5 * ((np.arange(size) - size // 2) / sigma) ** 2)
    g = np.outer(g, g)
    return tf.constant((g / g.sum())[:, :, None, None], tf.float32)


_SSIM_WIN = _gaussWindow()


def _ssimMap(a, b, maxVal=1.0):
    a, b = a[..., None], b[..., None]
    f = lambda t: tf.nn.conv2d(t, _SSIM_WIN, 1, 'VALID')
    ma, mb = f(a), f(b)
    saa = f(a * a) - ma * ma
    sbb = f(b * b) - mb * mb
    sab = f(a * b) - ma * mb
    c1, c2 = (0.01 * maxVal) ** 2, (0.03 * maxVal) ** 2
    return (((2 * ma * mb + c1) * (2 * sab + c2)) /
            ((ma * ma + mb * mb + c1) * (saa + sbb + c2)))[..., 0]


def l1Loss():
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        return tf.reduce_mean(tf.abs(y_pred[..., 0] - y_true[..., 0]))
    loss.__name__ = 'l1'
    return loss


def l1SsimLoss(ssimWeight=0.2):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        def perSample(args):
            yt, yp = args[0][..., 0], args[1][..., 0]
            return (tf.reduce_mean(tf.abs(yp - yt))
                    + ssimWeight * (1.0 - tf.reduce_mean(_ssimMap(yp, yt))))
        return tf.reduce_mean(
            tf.map_fn(perSample, (y_true, y_pred), fn_output_signature=tf.float32))
    loss.__name__ = 'l1_ssim'
    return loss


def buildLoss(spec, ssimWeight=0.2):
    spec = spec.upper().replace(' ', '')
    if spec == 'L1':
        return l1Loss()
    if spec in ('L1+SSIM', 'SSIM+L1'):
        return l1SsimLoss(ssimWeight)
    raise ValueError(f"unknown loss {spec!r}: use 'L1' or 'L1+SSIM'")


# ─── metric ───────────────────────────────────────────────────────────────────

def psnrMetric():
    def psnr(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        mse = tf.reduce_mean(tf.square(y_pred[..., 0] - y_true[..., 0]))
        return 10.0 * tf.math.log(1.0 / (mse + 1e-10)) / tf.math.log(10.0)
    psnr.__name__ = 'psnr'
    return psnr


# ─── training iterator ────────────────────────────────────────────────────────

class SRSequence(keras.utils.Sequence):
    def __init__(self, stream, batchSize, stepsPerEpoch, n_in=1):
        super().__init__()
        self.stream, self.batchSize, self.steps = stream, batchSize, stepsPerEpoch
        self.n_in = n_in
        self.lock = threading.Lock()

    def __len__(self):
        return self.steps

    def __getitem__(self, _):
        with self.lock:
            items = [self.stream.next() for _ in range(self.batchSize)]
        xs = np.stack([it[0] for it in items])
        if self.n_in != 1:
            xs = np.repeat(xs, self.n_in, axis=-1)
        ys = np.stack([it[1][..., None] for it in items])
        return xs, ys


# ─── inference ────────────────────────────────────────────────────────────────

def tileStarts(n, size, step):
    if n <= size:
        return [0]
    s = list(range(0, n - size, step))
    return s + [n - size]


def rawFileName(prefix, dtype, shapeZYX):
    """'<prefix>_<dtype>_<x>x<y>x<z>.raw'"""
    z, y, x = shapeZYX
    return f'{prefix}_{np.dtype(dtype).name}_{x}x{y}x{z}.raw'


def superResolveVolume(model, lrVol, lrSpacing, hrSpacing, lrNorm,
                       subvolSize=None, overlap=32, outFile=None, hrNorm=None,
                       outDtype='float32', batchSize=1, nPrefetch=4, verbose=True):
    """Super-resolve a full memory-mapped LR volume onto the HR grid.

    lrVol     : LR volume array or memmap (z,y,x)
    lrNorm    : (lo, hi) LR grey values mapped to 0..1 during training
    overlap   : subvolume overlap in HR voxels
    outFile   : write result to a .raw memmap; None keeps it in RAM
    hrNorm    : (lo, hi) to scale output back to HR grey values; None = keep 0..1
    Returns the output array (memmap if outFile given)."""
    if subvolSize is None:
        subvolSize = tuple(model.input_shape[1:4])
    n_in = model.input_shape[-1] or 1
    originUm = [0.0, 0.0, 0.0]
    outShape = tuple(int(v) for v in np.floor(np.array(lrVol.shape) * lrSpacing / hrSpacing))
    if verbose:
        print(f'HR output (z,y,x) {outShape}, '
              f'{np.prod(outShape) * np.dtype(outDtype).itemsize / 1e9:.2f} GB')
    out = (np.memmap(outFile, dtype=outDtype, mode='w+', shape=outShape) if outFile
           else np.zeros(outShape, outDtype))
    lo, hi = lrNorm
    margin = overlap // 2
    starts = list(itertools.product(
        *[tileStarts(n, s, s - overlap) for n, s in zip(outShape, subvolSize)]))

    infer = tf.function(model, input_signature=[
        tf.TensorSpec(shape=(None, *subvolSize, n_in), dtype=tf.float32)])

    def toOut(p):
        if hrNorm is not None:
            p = p * (hrNorm[1] - hrNorm[0]) + hrNorm[0]
        if np.issubdtype(np.dtype(outDtype), np.integer):
            info = np.iinfo(outDtype)
            p = np.clip(np.round(p), info.min, info.max)
        return p.astype(outDtype)

    def prepPatch(st):
        oUm = [originUm[i] + st[i] * hrSpacing for i in range(3)]
        c = srData.lrCoords(oUm, subvolSize, hrSpacing, lrSpacing)
        return (srData.sampleOnGrid(lrVol, (0, 0, 0), c) - lo) / (hi - lo)

    def flush(batch, pos):
        x = np.stack(batch)
        x = np.repeat(x[..., None], n_in, axis=-1)
        pred = infer(x).numpy()[..., 0]
        for p, st in zip(pred, pos):
            a0 = [s + (margin if s > 0 else 0) for s in st]
            a1 = [min(n, s + d - (margin if s + d < n else 0))
                  for s, d, n in zip(st, subvolSize, outShape)]
            sub = p[tuple(slice(a - s, b - s) for a, b, s in zip(a0, a1, st))]
            out[tuple(slice(a, b) for a, b in zip(a0, a1))] = toOut(sub)

    with ThreadPoolExecutor(max_workers=nPrefetch) as ex:
        for bStart in range(0, len(starts), batchSize):
            chunk = starts[bStart:bStart + batchSize]
            futures = [ex.submit(prepPatch, st) for st in chunk]
            batch = [f.result() for f in futures]
            flush(batch, chunk)
            if verbose and bStart % max(1, (100 // batchSize) * batchSize) == 0:
                print(f'  tile {bStart}/{len(starts)}')

    if outFile:
        out.flush()
    return out
