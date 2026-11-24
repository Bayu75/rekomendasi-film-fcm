"""Pengujian modul src/evaluation.py."""

import numpy as np
import pytest

from src.evaluation import (
    alpha_sweep,
    compare_methods,
    mean_precision_at_k,
    precision_at_k,
)


# Lima film, tiga genre: Action, Drama, Comedy.
# Film 0: Action, Comedy
# Film 1: Action
# Film 2: Drama
# Film 3: Action, Drama
# Film 4: Comedy
G_KECIL = np.array(
    [
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ],
    dtype=float,
)


@pytest.fixture
def data_acak():
    """30 film acak untuk tes agregat (seed 42)."""
    rng = np.random.default_rng(42)
    X = rng.random((30, 4))
    U = rng.random((30, 3))
    U = U / U.sum(axis=1, keepdims=True)
    G = np.zeros((30, 5), dtype=float)
    for i in range(30):
        n_g = int(rng.integers(1, 4))
        G[i, rng.choice(5, size=n_g, replace=False)] = 1.0
    return X, U, G


def test_precision_at_k_contoh_manual():
    """Contoh kecil dengan jawaban yang bisa dihitung tangan."""
    # Kueri film 0 (Action, Comedy). Irisan: 1=Action, 2=tidak, 4=Comedy.
    assert precision_at_k(0, [1, 2, 4], G_KECIL) == pytest.approx(2 / 3)
    # Ketiganya relevan (Action, Action, Comedy).
    assert precision_at_k(0, [1, 3, 4], G_KECIL) == pytest.approx(1.0)
    # Kueri film 2 (Drama). Hanya film 3 yang berbagi genre.
    assert precision_at_k(2, [0, 1, 4], G_KECIL) == pytest.approx(0.0)
    assert precision_at_k(2, [3, 0], G_KECIL) == pytest.approx(0.5)
    # min_shared=2: tidak ada film lain yang berbagi dua genre dengan film 0.
    assert precision_at_k(0, [1, 3, 4], G_KECIL, min_shared=2) == pytest.approx(0.0)


def test_compare_methods_baseline_sama_dengan_alpha_1(data_acak):
    """precision_baseline harus sama dengan mean P@K pada α = 1."""
    X, U, G = data_acak
    hasil = compare_methods(X, U, G, k=5, alpha=0.5, seed=42)
    p_baseline = mean_precision_at_k(X, U, G, k=5, alpha=1.0)
    assert hasil["precision_baseline"] == pytest.approx(p_baseline)
    assert hasil["k"] == 5
    assert hasil["alpha"] == 0.5
    assert hasil["selisih"] == pytest.approx(
        hasil["precision_fuzzy"] - hasil["precision_baseline"]
    )

    # Jika alpha fuzzy juga 1, kedua metode identik.
    sama = compare_methods(X, U, G, k=5, alpha=1.0)
    assert sama["precision_fuzzy"] == pytest.approx(sama["precision_baseline"])
    assert sama["selisih"] == pytest.approx(0.0)


def test_alpha_sweep_panjang_dan_rentang(data_acak):
    """Hasil sweep berjumlah sama dengan daftar α, dan P@K berada di [0, 1]."""
    X, U, G = data_acak
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    hasil = alpha_sweep(X, U, G, alphas=alphas, k=5)
    assert len(hasil) == len(alphas)
    for a, baris in zip(alphas, hasil):
        assert baris["alpha"] == pytest.approx(a)
        assert 0.0 <= baris["precision"] <= 1.0


def test_bentuk_tidak_valid_memunculkan_valueerror(data_acak):
    """Bentuk matriks yang tidak cocok, k di luar rentang, atau α salah harus gagal."""
    X, U, G = data_acak
    with pytest.raises(ValueError, match="Jumlah film"):
        mean_precision_at_k(X, U, G[:-1], k=5)
    with pytest.raises(ValueError, match="Jumlah film"):
        mean_precision_at_k(X, U[:-1], G, k=5)
    with pytest.raises(ValueError, match="2 dimensi"):
        precision_at_k(0, [1, 2], G[0])
    with pytest.raises(ValueError, match="k harus"):
        mean_precision_at_k(X, U, G, k=0)
    with pytest.raises(ValueError, match="k harus"):
        mean_precision_at_k(X, U, G, k=30)
    with pytest.raises(ValueError, match="alpha harus"):
        mean_precision_at_k(X, U, G, k=5, alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        alpha_sweep(X, U, G, alphas=(0.0, 1.5), k=5)


def test_hasil_reproduktif_dengan_seed(data_acak):
    """Seed yang sama menghasilkan P@K yang sama; seed berbeda boleh berbeda."""
    X, U, G = data_acak
    p1 = mean_precision_at_k(X, U, G, k=5, alpha=0.5, n_queries=10, seed=42)
    p2 = mean_precision_at_k(X, U, G, k=5, alpha=0.5, n_queries=10, seed=42)
    p3 = mean_precision_at_k(X, U, G, k=5, alpha=0.5, n_queries=10, seed=0)
    assert p1 == pytest.approx(p2)
    assert 0.0 <= p1 <= 1.0
    assert 0.0 <= p3 <= 1.0

    h1 = compare_methods(X, U, G, k=5, alpha=0.5, n_queries=10, seed=42)
    h2 = compare_methods(X, U, G, k=5, alpha=0.5, n_queries=10, seed=42)
    assert h1["precision_fuzzy"] == pytest.approx(h2["precision_fuzzy"])
    assert h1["precision_baseline"] == pytest.approx(h2["precision_baseline"])
