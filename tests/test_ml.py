"""Tests for the learned real-bogus vetter (Phase 2)."""

import numpy as np
import pytest

from astrohunt.catalog import load_reference_catalog
from astrohunt.mlvet import RealBogusModel, build_dataset
from astrohunt.pipeline import run_pipeline
from astrohunt.synthetic import SyntheticConfig, generate


@pytest.fixture(scope="module")
def dataset():
    X, y = build_dataset(n_sets=3, n_cosmic_rays=20)
    return X, y


def test_dataset_has_both_classes(dataset):
    X, y = dataset
    assert len(X) > 200
    assert y.sum() > 0 and (1 - y).sum() > 0   # real and bogus both present


def test_mlp_classifies(dataset):
    X, y = dataset
    rng = np.random.default_rng(0)
    p = rng.permutation(len(X))
    X, y = X[p], y[p]
    ntr = int(0.8 * len(X))
    m = RealBogusModel(backend="mlp").fit(X[:ntr], y[:ntr])
    acc = ((m.predict_proba(X[ntr:]) >= 0.5).astype(int) == y[ntr:]).mean()
    assert acc > 0.9


def test_save_load_roundtrip(dataset, tmp_path):
    X, y = dataset
    m = RealBogusModel(backend="mlp").fit(X, y)
    path = str(tmp_path / "vetter.pkl")
    m.save(path)
    m2 = RealBogusModel.load(path)
    assert np.allclose(m.predict_proba(X[:20]), m2.predict_proba(X[:20]))


def test_pipeline_with_ml_vetter(dataset, tmp_path):
    X, y = dataset
    model = RealBogusModel(backend="mlp").fit(X, y)
    mpath = str(tmp_path / "vetter.pkl")
    model.save(mpath)

    paths, truth_path = generate(
        SyntheticConfig(seed=42, n_cosmic_rays=10), str(tmp_path / "set")
    )
    refcat = load_reference_catalog(truth_path)
    result = run_pipeline(
        paths, reference_catalog=refcat, use_online=False,
        vetter="ml", vet_model=mpath,
    )
    assert result.n_candidates >= 1
    assert result.median_rms_arcsec < 1.0


def test_cnn_backend_if_torch():
    torch = pytest.importorskip("torch")  # noqa: F841
    X, y = build_dataset(n_sets=2, n_cosmic_rays=20)
    m = RealBogusModel(backend="cnn", epochs=8).fit(X, y)
    probs = m.predict_proba(X[:32])
    assert probs.shape == (32,)
    assert ((probs >= 0) & (probs <= 1)).all()
