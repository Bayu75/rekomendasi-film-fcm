"""Pengujian unit untuk src/fcm.py. Jalankan: python -m pytest tests -v"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from src.fcm import (fcm, fpc, init_membership, membership, select_c,
                     update_membership)


@pytest.fixture(scope="module")
def blobs():
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.8, random_state=42)
    return MinMaxScaler().fit_transform(X)


@pytest.fixture(scope="module")
def hasil(blobs):
    return fcm(blobs, 3, error=1e-8, maxiter=1000)


def test_bentuk_keluaran(blobs, hasil):
    assert set(hasil) == {"V", "U", "J_history", "n_iter", "fpc"}
    assert hasil["U"].shape == (300, 3)
    assert hasil["V"].shape == (3, 2)
    assert isinstance(hasil["J_history"], list)
    assert len(hasil["J_history"]) == hasil["n_iter"] >= 1
    assert isinstance(hasil["fpc"], float)


def test_baris_U_berjumlah_satu(hasil):
    np.testing.assert_allclose(hasil["U"].sum(axis=1), 1.0, atol=1e-12)


def test_U_dalam_rentang_0_1(hasil):
    assert np.all(hasil["U"] >= 0) and np.all(hasil["U"] <= 1)


def test_init_membership_valid():
    U = init_membership(10, 4, seed=1)
    assert U.shape == (10, 4)
    np.testing.assert_allclose(U.sum(axis=1), 1.0)


def test_seed_sama_hasil_identik(blobs):
    a = fcm(blobs, 3, seed=7)
    b = fcm(blobs, 3, seed=7)
    np.testing.assert_array_equal(a["U"], b["U"])
    np.testing.assert_array_equal(a["V"], b["V"])
    assert a["J_history"] == b["J_history"]


def test_seed_berbeda_U_awal_berbeda():
    # Seed berbeda menghasilkan U awal berbeda; hasil akhir boleh sama atau
    # berbeda (urutan cluster bisa tertukar), jadi yang diuji adalah U awal.
    assert not np.allclose(init_membership(50, 3, seed=1), init_membership(50, 3, seed=2))


def test_seed_berbeda_cluster_tetap_sama_setelah_dicocokkan(blobs):
    a = fcm(blobs, 3, seed=1, error=1e-8)
    b = fcm(blobs, 3, seed=2, error=1e-8)
    # Pusat yang sama (mungkin urutannya berbeda) untuk data yang jelas terpisah.
    Va = a["V"][np.lexsort(a["V"].T[::-1])]
    Vb = b["V"][np.lexsort(b["V"].T[::-1])]
    np.testing.assert_allclose(Va, Vb, atol=1e-6)


@pytest.mark.parametrize("c", [2, 3, 5])
def test_fpc_dalam_rentang(blobs, c):
    r = fcm(blobs, c)
    assert 1.0 / c - 1e-12 <= r["fpc"] <= 1.0 + 1e-12
    assert r["fpc"] == pytest.approx(fpc(r["U"]))


def test_fpc_batas_teori():
    assert fpc(np.full((4, 2), 0.5)) == pytest.approx(0.5)      # paling kabur = 1/c
    assert fpc(np.eye(3)) == pytest.approx(1.0)                  # paling tegas = 1


def test_J_history_tidak_naik(blobs):
    for X, c in [(blobs, 3), (blobs, 5)]:
        J = np.array(fcm(X, c, error=1e-9, maxiter=500)["J_history"])
        assert np.all(np.diff(J) <= 1e-10 * max(1.0, J[0]))


def test_membership_titik_latih_sama_dengan_U(blobs, hasil):
    U_pred = membership(blobs, hasil["V"])
    np.testing.assert_allclose(U_pred, hasil["U"], atol=1e-10)
    u0 = membership(blobs[0], hasil["V"])
    assert u0.shape == (3,)
    np.testing.assert_allclose(u0, hasil["U"][0], atol=1e-10)


def test_membership_titik_tepat_di_pusat():
    V = np.array([[0.0, 0.0], [1.0, 1.0]])
    u = membership(np.array([1.0, 1.0]), V)
    np.testing.assert_array_equal(u, [0.0, 1.0])


def test_update_membership_jarak_nol_aman():
    D = np.array([[0.0, 2.0, 3.0], [1.0, 1.0, 2.0], [0.0, 0.0, 5.0]])
    U = update_membership(D)
    assert np.all(np.isfinite(U))
    np.testing.assert_allclose(U[0], [1, 0, 0])
    np.testing.assert_allclose(U[2], [0.5, 0.5, 0])
    np.testing.assert_allclose(U.sum(axis=1), 1.0)


def test_cocok_dengan_skfuzzy(blobs):
    skfuzzy = pytest.importorskip("skfuzzy")
    U0 = init_membership(blobs.shape[0], 3, seed=42)
    ours = fcm(blobs, 3, error=1e-8, maxiter=1000, init=U0)
    cntr, u, _, _, jm, _, fpc_sk = skfuzzy.cluster.cmeans(
        blobs.T, 3, 2.0, error=1e-8, maxiter=1000, init=U0.T)
    np.testing.assert_allclose(ours["V"], cntr, atol=1e-6)
    np.testing.assert_allclose(ours["U"], u.T, atol=1e-6)
    assert ours["fpc"] == pytest.approx(fpc_sk, abs=1e-8)
    assert ours["J_history"][-1] == pytest.approx(jm[-1], rel=1e-8)


def test_select_c(blobs):
    df = select_c(blobs, range(2, 6), seed=42)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["c", "fpc", "silhouette"]
    assert df["c"].tolist() == [2, 3, 4, 5]
    # Data dibangkitkan dengan 3 cluster -> c=3 terbaik menurut kedua metrik.
    assert df.loc[df["fpc"].idxmax(), "c"] == 3
    assert df.loc[df["silhouette"].idxmax(), "c"] == 3


@pytest.mark.parametrize("kwargs, pesan", [
    (dict(X=np.zeros((5, 2)), c=1), "minimal 2"),
    (dict(X=np.zeros((5, 2)), c=6), "melebihi"),
    (dict(X=np.zeros((5, 2)), c=2.5), "bilangan bulat"),
    (dict(X=np.zeros((5, 2)), c=2, m=1.0), "lebih besar dari 1"),
    (dict(X=np.zeros((5, 2)), c=2, m=0.5), "lebih besar dari 1"),
    (dict(X=np.zeros(5), c=2), "2 dimensi"),
    (dict(X=np.array([[1.0, np.nan], [0, 0], [1, 1]]), c=2), "NaN"),
    (dict(X=np.array([[1.0, np.inf], [0, 0], [1, 1]]), c=2), "NaN"),
    (dict(X=[["a", "b"], ["c", "d"]], c=2), "numerik"),
    (dict(X=np.zeros((5, 2)), c=2, error=0), "error"),
    (dict(X=np.zeros((5, 2)), c=2, maxiter=0), "maxiter"),
    (dict(X=np.zeros((5, 2)), c=2, init=np.ones((4, 2))), "init"),
])
def test_input_tidak_valid(kwargs, pesan):
    with pytest.raises(ValueError, match=pesan):
        fcm(**kwargs)


def test_membership_fitur_tidak_cocok():
    with pytest.raises(ValueError, match="fitur"):
        membership(np.zeros(3), np.zeros((2, 2)))


def test_semua_data_identik_tidak_error():
    # Semua data sama -> jarak ke pusat nol; harus tetap berjalan tanpa NaN.
    r = fcm(np.ones((6, 2)), 2)
    assert np.all(np.isfinite(r["U"])) and np.all(np.isfinite(r["V"]))
    np.testing.assert_allclose(r["U"].sum(axis=1), 1.0)
