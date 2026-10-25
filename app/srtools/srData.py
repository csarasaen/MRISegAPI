"""Data utilities for the µCT/MRI super-resolution pipeline — inference-only build.

Adapted from sr_pipeline_v04/srtools/srData.py.
Change: the `fileIO` dependency (training-time utility) is replaced with a
        self-contained regex parser for the .raw filename convention:
        <prefix>_<dtype>_<x>x<y>x<z>.raw  (x,y,z are voxel counts per axis)
"""
# ruff: noqa: N999
import itertools
import json
import math
import multiprocessing as mp
import os
import re
from collections import deque

import numpy as np
from scipy import ndimage


# ─── filename parser (replaces fileIO dependency) ─────────────────────────────

_RAW_RE = re.compile(r'_(u?int\d+|float\d+)_(\d+)x(\d+)x(\d+)\.raw$', re.IGNORECASE)


def parseRawFilename(fileName):
    """Parse dims (x,y,z) and dtype from '<prefix>_<dtype>_<x>x<y>x<z>.raw'.
    Returns (dtype_str, (x, y, z)) or raises ValueError."""
    m = _RAW_RE.search(os.path.basename(fileName))
    if not m:
        raise ValueError(
            f"Cannot parse dims/dtype from filename: {os.path.basename(fileName)!r}\n"
            "Expected format: name_float32_200x200x100.raw"
        )
    dtype = m.group(1)
    x, y, z = int(m.group(2)), int(m.group(3)), int(m.group(4))
    return dtype, (x, y, z)


# ─── volume I/O ───────────────────────────────────────────────────────────────

def mapRaw(fileName, dims=None, dtype=None):
    """Memory-map a .raw volume read-only, shape (z, y, x).
    dims (x,y,z) and dtype are parsed from the file name unless given explicitly."""
    if dims is None or dtype is None:
        fType, fDims = parseRawFilename(fileName)
        dims  = fDims  if dims  is None else dims
        dtype = fType  if dtype is None else dtype
    shape = tuple(int(d) for d in np.asarray(dims)[::-1])   # (z, y, x)
    return np.memmap(fileName, dtype=np.dtype(str(dtype)), mode='r', shape=shape)


def readPadded(vol, start, stop):
    """vol[start:stop] as float32; out-of-bounds regions are edge-padded."""
    lo = [max(0, s) for s in start]
    hi = [min(n, e) for n, e in zip(vol.shape, stop)]
    if any(h <= l for l, h in zip(lo, hi)):
        raise ValueError('block lies completely outside the volume')
    block = np.asarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
    pad = [(l - s, e - h) for s, e, l, h in zip(start, stop, lo, hi)]
    if any(p != (0, 0) for p in pad):
        block = np.pad(block, pad, mode='edge')
    return block


def robustPercentiles(vol, p=(0.5, 99.9), nBlocks=64, block=48, seed=0):
    """Grey-value percentiles from random blocks (never loads the full volume)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nBlocks):
        o = [int(rng.integers(0, max(1, s - block))) for s in vol.shape]
        vals.append(np.asarray(
            vol[o[0]:o[0]+block, o[1]:o[1]+block, o[2]:o[2]+block],
            dtype=np.float32).ravel())
    return [float(v) for v in np.percentile(np.concatenate(vals), p)]


# ─── coordinate helpers ───────────────────────────────────────────────────────

def lrCoords(originUm, size, hrSpacing, lrSpacing, lrOffsetUm=(0, 0, 0)):
    """LR-index coordinates of HR subvolume voxels, one array per axis."""
    return [(originUm[a] + np.arange(size[a]) * hrSpacing - lrOffsetUm[a]) / lrSpacing
            for a in range(3)]


def sampleOnGrid(src, srcOriginInd, coords1d, order=3):
    """Interpolate src at separable grid coords1d. Only reads the needed block."""
    lo = [math.floor(c[0]) - 2 for c in coords1d]
    hi = [math.ceil(c[-1]) + 3 for c in coords1d]
    block = readPadded(src, [l - o for l, o in zip(lo, srcOriginInd)],
                       [h - o for h, o in zip(hi, srcOriginInd)])
    grid = np.meshgrid(*[c - l for c, l in zip(coords1d, lo)], indexing='ij')
    return ndimage.map_coordinates(block, grid, order=order, mode='nearest').astype(np.float32)


# ─── normalisation stats ──────────────────────────────────────────────────────

def prepareStats(rois, statsFile):
    """Normalisation percentiles per ROI, cached in a json file."""
    try:
        with open(statsFile) as f:
            stats = json.load(f)
    except FileNotFoundError:
        stats = {}
    changed = False
    for roi in rois:
        key = roi['hr']
        if key in stats:
            continue
        hr, lr = mapRaw(roi['hr']), mapRaw(roi['lr'])
        stats[key] = {'hrNorm': robustPercentiles(hr), 'lrNorm': robustPercentiles(lr)}
        changed = True
        print(f'[stats] {os.path.basename(key)}: {stats[key]}')
    if changed:
        with open(statsFile, 'w') as f:
            json.dump(stats, f, indent=2)
    return stats


# ─── ROI pair (training) ──────────────────────────────────────────────────────

class ROIPair:
    """One HR/LR pair (both memory-mapped)."""

    def __init__(self, roi, stats, hrSpacing, lrSpacing, lrOffset=(0, 0, 0)):
        self.hr = mapRaw(roi['hr'])
        self.lr = mapRaw(roi['lr'])
        st = stats[roi['hr']]
        self.hrLo, self.hrHi = st['hrNorm']
        self.lrLo, self.lrHi = st['lrNorm']
        self.hrSp = float(hrSpacing)
        self.lrSp = float(lrSpacing)
        self.lrOff = [float(v) for v in lrOffset]
        lrExt = [n * self.lrSp + o for n, o in zip(self.lr.shape, self.lrOff)]
        self.validHR = [min(h, int(e / self.hrSp)) for h, e in zip(self.hr.shape, lrExt)]

    def hrPatch(self, origin, dims):
        return ((readPadded(self.hr, origin, [o + d for o, d in zip(origin, dims)])
                 - self.hrLo) / (self.hrHi - self.hrLo))

    def lrInput(self, origin, dims):
        c = lrCoords([o * self.hrSp for o in origin], dims, self.hrSp, self.lrSp, self.lrOff)
        raw = sampleOnGrid(self.lr, (0, 0, 0), c)
        return (raw - self.lrLo) / (self.lrHi - self.lrLo)


def _augment(x, y, rng):
    """Random flips / 90° rotations / axis permutations."""
    arrays = [x, y]
    shape = x.shape
    for ax in range(3):
        if rng.random() < 0.5:
            arrays = [np.flip(a, ax) for a in arrays]
    if shape[1] == shape[2]:
        k = int(rng.integers(0, 4))
        arrays = [np.rot90(a, k, (1, 2)) for a in arrays]
    if shape[0] == shape[1] == shape[2]:
        p = rng.permutation(3)
        arrays = [a.transpose(p) for a in arrays]
    return [np.ascontiguousarray(a) for a in arrays]


class SubvolSampler:
    """Draws random (LR_input, HR_target) subvolume pairs from a list of ROIs."""

    def __init__(self, rois, stats, dims, hrSpacing, lrSpacing, lrOffset=(0, 0, 0), augment=True):
        self.dims = tuple(int(d) for d in dims)
        self.augment = augment
        self.pairs = [ROIPair(r, stats, hrSpacing, lrSpacing, lrOffset) for r in rois]
        vol = np.array([np.prod([max(1, v - d) for v, d in zip(p.validHR, self.dims)])
                        for p in self.pairs], float)
        self.pRoi = vol / vol.sum()

    def sample(self, rng):
        pair = self.pairs[rng.choice(len(self.pairs), p=self.pRoi)]
        origin = [int(rng.integers(0, max(1, v - d))) for v, d in zip(pair.validHR, self.dims)]
        x = pair.lrInput(origin, self.dims)
        y = pair.hrPatch(origin, self.dims)
        if self.augment:
            x, y = _augment(x, y, rng)
        return x[..., None], y.astype(np.float32)


# ─── parallel stream ──────────────────────────────────────────────────────────

_WORKER = {}


def _initWorker(rois, stats, dims, hrSpacing, lrSpacing, lrOffset, augment):
    _WORKER['sampler'] = SubvolSampler(rois, stats, dims, hrSpacing, lrSpacing, lrOffset, augment)


def _work(seed):
    return _WORKER['sampler'].sample(np.random.default_rng(seed))


class SubvolStream:
    """Endless stream of (x, y) pairs produced by worker processes."""

    def __init__(self, rois, stats, dims, hrSpacing, lrSpacing, lrOffset=(0, 0, 0),
                 nWorkers=4, prefetch=16, seed=0, augment=True):
        self._args = (rois, stats, dims, hrSpacing, lrSpacing, lrOffset, augment)
        self.nWorkers, self.prefetch = nWorkers, prefetch
        self.seeds = itertools.count(seed * 1_000_000_007)
        self.pool, self.queue = None, deque()

    def start(self):
        if self.nWorkers > 0 and self.pool is None:
            self.pool = mp.get_context('fork').Pool(self.nWorkers, _initWorker, self._args)
        elif self.nWorkers <= 0:
            _initWorker(*self._args)
        return self

    def next(self):
        if self.nWorkers <= 0:
            if not _WORKER:
                self.start()
            return _work(next(self.seeds))
        if self.pool is None:
            self.start()
        while len(self.queue) < self.prefetch:
            self.queue.append(self.pool.apply_async(_work, (next(self.seeds),)))
        res = self.queue.popleft().get()
        self.queue.append(self.pool.apply_async(_work, (next(self.seeds),)))
        return res

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool = None
