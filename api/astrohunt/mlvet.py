"""Learned real-bogus vetting (the Phase-2 upgrade to rule-based vetting).

Trains a classifier to separate real point sources from artifacts using
postage-stamp cutouts auto-labelled against the known catalogue (SkyBoT/truth) —
exactly the bootstrap the research recommended. Two backends:

  * "mlp" (default): a small from-scratch NumPy multilayer perceptron. Zero extra
    dependencies, CPU-only, trains in seconds — runs anywhere astrohunt does.
  * "cnn" (optional): a small PyTorch convolutional net, used if torch is
    installed. Same interface, drop-in.

Both consume the same feature vector (a normalised cutout + three morphological
summaries), so the pipeline integration is backend-agnostic.
"""

from __future__ import annotations

import json
import pickle
import tempfile

import numpy as np

CUTOUT = 21          # postage-stamp side length (pixels)
_HALF = CUTOUT // 2


# --- cutouts & features -----------------------------------------------------

def extract_cutout(data, x, y, size=CUTOUT):
    """Return a background-subtracted, std-normalised cutout, or None at edges."""
    half = size // 2
    xi = int(round(x)) - half
    yi = int(round(y)) - half
    if xi < 0 or yi < 0 or xi + size > data.shape[1] or yi + size > data.shape[0]:
        return None
    c = data[yi:yi + size, xi:xi + size].astype(float)
    c = c - np.median(c)
    s = c.std()
    return c / s if s > 0 else c


def _morpho(c):
    """Three cheap shape summaries from a cutout: sharpness, fwhm, elongation."""
    cc = np.clip(c, 0, None)
    tot = cc.sum()
    if tot <= 0:
        return np.array([0.0, 0.0, 1.0])
    ny, nx = cc.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    xc = (xx * cc).sum() / tot
    yc = (yy * cc).sum() / tot
    mxx = ((xx - xc) ** 2 * cc).sum() / tot
    myy = ((yy - yc) ** 2 * cc).sum() / tot
    mxy = ((xx - xc) * (yy - yc) * cc).sum() / tot
    tr = mxx + myy
    disc = max(tr ** 2 / 4 - (mxx * myy - mxy ** 2), 0.0)
    l1 = max(tr / 2 + np.sqrt(disc), 1e-6)
    l2 = max(tr / 2 - np.sqrt(disc), 1e-6)
    sharp = c.max() / tot
    fwhm = 2.3548 * np.sqrt(max(tr / 2.0, 1e-6))
    return np.array([sharp, fwhm, np.sqrt(l1 / l2)])


def features_from_cutout(c):
    return np.concatenate([c.ravel(), _morpho(c)])


N_FEATURES = CUTOUT * CUTOUT + 3


# --- NumPy MLP backend ------------------------------------------------------

class NumpyMLP:
    """A tiny 1-hidden-layer MLP trained with mini-batch SGD + cross-entropy."""

    def __init__(self, n_in=N_FEATURES, n_hidden=48, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.08, (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.08, (n_hidden, 1))
        self.b2 = np.zeros(1)
        self.lr = lr
        self.mu = None
        self.sd = None

    def _std(self, X, fit=False):
        if fit:
            self.mu = X.mean(0)
            self.sd = X.std(0) + 1e-6
        return (X - self.mu) / self.sd

    def fit(self, X, y, epochs=250, batch=64, seed=0):
        X = self._std(np.asarray(X, float), fit=True)
        y = np.asarray(y, float).reshape(-1, 1)
        rng = np.random.default_rng(seed)
        n = len(X)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                xb, yb = X[b], y[b]
                a1 = np.tanh(xb @ self.W1 + self.b1)
                p = 1.0 / (1.0 + np.exp(-(a1 @ self.W2 + self.b2)))
                d2 = (p - yb) / len(b)
                dW2 = a1.T @ d2
                db2 = d2.sum(0)
                d1 = (d2 @ self.W2.T) * (1 - a1 ** 2)
                dW1 = xb.T @ d1
                db1 = d1.sum(0)
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
        return self

    def predict_proba(self, X):
        X = self._std(np.asarray(X, float))
        a1 = np.tanh(X @ self.W1 + self.b1)
        return (1.0 / (1.0 + np.exp(-(a1 @ self.W2 + self.b2)))).ravel()


# --- optional PyTorch CNN backend ------------------------------------------

def _build_torch_cnn():
    import torch
    from torch import nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
                nn.Flatten(), nn.Linear(16 * 16, 32), nn.ReLU(), nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x)

    return CNN()


class TorchCNN:
    def __init__(self, size=CUTOUT, lr=1e-3, epochs=15, seed=0):
        import torch

        torch.manual_seed(seed)
        self.size = size
        self.lr = lr
        self.epochs = epochs
        self.model = _build_torch_cnn()

    def _imgs(self, X):
        import torch

        X = np.asarray(X, float)
        imgs = X[:, : self.size * self.size].reshape(-1, 1, self.size, self.size)
        return torch.tensor(imgs, dtype=torch.float32)

    def fit(self, X, y, batch=64):
        import torch
        from torch import nn

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        imgs = self._imgs(X)
        yt = torch.tensor(np.asarray(y, float).reshape(-1, 1), dtype=torch.float32)
        n = len(imgs)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch):
                b = perm[s:s + batch]
                opt.zero_grad()
                loss = loss_fn(self.model(imgs[b]), yt[b])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._imgs(X)).numpy().ravel()
        return 1.0 / (1.0 + np.exp(-logits))


# --- unified model wrapper --------------------------------------------------

class RealBogusModel:
    def __init__(self, backend="mlp", **kw):
        if backend == "cnn":
            self.impl = TorchCNN(**kw)
        elif backend == "mlp":
            self.impl = NumpyMLP(**kw)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend

    def fit(self, X, y, **kw):
        self.impl.fit(X, y, **kw)
        return self

    def predict_proba(self, X):
        return self.impl.predict_proba(X)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"backend": self.backend, "impl": self.impl}, f)
        return path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls.__new__(cls)
        m.backend = d["backend"]
        m.impl = d["impl"]
        return m


# --- dataset construction & inference --------------------------------------

def _label_detections(srcs, wcs, truth, frame_index, tol_px=2.5):
    xs = np.asarray(srcs["x"], float)
    ys = np.asarray(srcs["y"], float)
    sra = [s["ra"] for s in truth["stars"]]
    sdec = [s["dec"] for s in truth["stars"]]
    sx, sy = wcs.world_to_pixel_values(sra, sdec)
    pts = list(zip(np.atleast_1d(sx), np.atleast_1d(sy)))
    for fr in truth["asteroid"]["frames"]:
        if fr["frame"] == frame_index and fr["in_frame"]:
            pts.append((fr["x"], fr["y"]))
    pts = np.array(pts)
    labels = np.zeros(len(xs), int)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if np.hypot(pts[:, 0] - x, pts[:, 1] - y).min() <= tol_px:
            labels[i] = 1
    return labels


def build_dataset(n_sets=5, n_cosmic_rays=25, seed0=100):
    """Auto-labelled (features, labels) from synthetic frames with artifacts."""
    from .extract import extract_sources
    from .io import load_frame
    from .synthetic import SyntheticConfig, generate

    X, Y = [], []
    for k in range(n_sets):
        with tempfile.TemporaryDirectory() as td:
            cfg = SyntheticConfig(seed=seed0 + k, n_cosmic_rays=n_cosmic_rays)
            paths, truth_path = generate(cfg, td)
            truth = json.load(open(truth_path))
            for i, p in enumerate(paths):
                frame = load_frame(p, index=i)
                srcs = extract_sources(frame.data, fwhm=cfg.fwhm_px, threshold_sigma=5.0)
                labels = _label_detections(srcs, frame.wcs, truth, i)
                for j in range(len(srcs)):
                    c = extract_cutout(frame.data, srcs["x"][j], srcs["y"][j])
                    if c is not None:
                        X.append(features_from_cutout(c))
                        Y.append(int(labels[j]))
    return np.array(X), np.array(Y)


def predict_mask(model: RealBogusModel, data, srcs, thresh=0.5):
    """Return a boolean 'is-real' mask for the sources in `srcs`."""
    feats, idx = [], []
    for j in range(len(srcs)):
        c = extract_cutout(data, srcs["x"][j], srcs["y"][j])
        if c is not None:
            feats.append(features_from_cutout(c))
            idx.append(j)
    mask = np.ones(len(srcs), bool)   # edge sources we can't cut out -> keep
    if feats:
        probs = model.predict_proba(np.array(feats))
        for j, p in zip(idx, probs):
            mask[j] = p >= thresh
    return mask
