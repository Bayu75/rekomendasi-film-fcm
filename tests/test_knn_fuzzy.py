"""Pengujian modul src/knn_fuzzy.py."""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from src.knn_fuzzy import (
    combined_distance,
    membership_new,
    recommend,
    recommend_new,
)


@pytest.fixture
def data_dummy():
    """20 film, 4 fitur, dan keanggotaan 3 cluster (seed 42)."""
    rng = np.random.default_rng(42)
    X = rng.random((20, 4))
    U = rng.random((20, 3))
    U = U / U.sum(axis=1, keepdims=True)
    return X, U


def test_alpha_1_cocok_dengan_sklearn(data_dummy):
    """Top-k pada α = 1 harus sama dengan NearestNeighbors (hanya fitur)."""
    X, U = data_dummy
    query_idx, k = 0, 5

    indeks, jarak = recommend(X, U, query_idx, k=k, alpha=1.0)

    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    model.fit(X)
    jarak_sk, indeks_sk = model.kneighbors(X[[query_idx]])
    # Tetangga pertama pasti film kueri itu sendiri, jadi dilewati.
    assert indeks_sk[0, 0] == query_idx
    np.testing.assert_array_equal(indeks, indeks_sk[0, 1:])
    np.testing.assert_allclose(jarak, jarak_sk[0, 1:])


def test_query_tidak_masuk_hasil(data_dummy):
    """Film yang dijadikan kueri tidak boleh muncul di rekomendasi."""
    X, U = data_dummy
    for query_idx in (0, 7, 19):
        indeks, _ = recommend(X, U, query_idx, k=5, alpha=0.5)
        assert query_idx not in indeks
        assert len(indeks) == 5


def test_membership_new_jumlah_satu_dan_nonnegatif():
    """Setiap vektor keanggotaan bernilai ≥ 0 dan jumlahnya 1."""
    rng = np.random.default_rng(42)
    V = rng.random((5, 4))
    for _ in range(10):
        u = membership_new(rng.random(4), V, m=2.0)
        assert u.shape == (5,)
        assert np.all(u >= 0)
        np.testing.assert_allclose(u.sum(), 1.0)


def test_tepat_di_pusat_hasilnya_one_hot():
    """Film yang berimpit dengan pusat cluster mendapat keanggotaan one-hot."""
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    u = membership_new(V[1], V, m=2.0)
    np.testing.assert_allclose(u, np.array([0.0, 1.0, 0.0]))
    assert np.all(u >= 0)
    np.testing.assert_allclose(u.sum(), 1.0)


def test_k_dan_alpha_tidak_valid(data_dummy):
    """k di luar 1..n-1 atau alpha di luar [0, 1] harus memunculkan ValueError."""
    X, U = data_dummy
    n = X.shape[0]

    with pytest.raises(ValueError, match="k harus"):
        recommend(X, U, 0, k=0, alpha=0.5)
    with pytest.raises(ValueError, match="k harus"):
        recommend(X, U, 0, k=n, alpha=0.5)
    with pytest.raises(ValueError, match="k harus"):
        recommend(X, U, 0, k=-1, alpha=0.5)
    with pytest.raises(ValueError, match="alpha harus"):
        recommend(X, U, 0, k=5, alpha=1.5)
    with pytest.raises(ValueError, match="alpha harus"):
        combined_distance(X, U, X[0], U[0], alpha=-0.1)


def test_alpha_0_hanya_memakai_keanggotaan(data_dummy):
    """Pada α = 0, jarak gabungan sama dengan jarak Euclidean pada U saja."""
    X, U = data_dummy
    query_idx, k = 3, 5

    jarak_semua = combined_distance(X, U, X[query_idx], U[query_idx], alpha=0.0)
    jarak_U = np.linalg.norm(U - U[query_idx], axis=1)
    np.testing.assert_allclose(jarak_semua, jarak_U)

    indeks, jarak = recommend(X, U, query_idx, k=k, alpha=0.0)
    jarak_U = jarak_U.copy()
    jarak_U[query_idx] = np.inf
    expected = np.argsort(jarak_U)[:k]
    np.testing.assert_array_equal(indeks, expected)
    np.testing.assert_allclose(jarak, jarak_U[expected])


def test_recommend_new_mengembalikan_k_hasil(data_dummy):
    """Film baru dapat direkomendasikan setelah keanggotaannya dihitung dari V."""
    X, U = data_dummy
    rng = np.random.default_rng(0)
    V = rng.random((U.shape[1], X.shape[1]))
    x_new = rng.random(X.shape[1])

    u_new = membership_new(x_new, V)
    np.testing.assert_allclose(u_new.sum(), 1.0)

    indeks, jarak = recommend_new(x_new, X, U, V, k=5, alpha=0.5, m=2.0)
    assert len(indeks) == 5
    assert np.all(np.diff(jarak) >= -1e-12)  # sudah terurut menaik
