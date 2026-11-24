"""Tes contoh perhitungan, uraian jarak integrasi FCM→KNN, prediksi rating, dan jalur error."""
import numpy as np
import pandas as pd
import pytest

from src import contoh, demo_data, interpret, plots


def test_contoh_fcm_properti():
    r = contoh.langkah_fcm(contoh.X_CONTOH, contoh.U0_CONTOH, m=2.0, n_iter=4)
    for L in r:
        assert np.allclose(L["U_out"].sum(axis=1), 1.0)           # tiap baris U berjumlah 1
    J = [L["J"] for L in r]
    assert all(b <= a + 1e-12 for a, b in zip(J, J[1:]))           # J_m tidak naik
    assert r[1]["delta"] < r[0]["delta"]                           # perubahan U mengecil


def test_contoh_pusat_cocok_rumus_tangan():
    r = contoh.langkah_fcm(contoh.X_CONTOH, contoh.U0_CONTOH, m=2.0, n_iter=1)[0]
    U, X = r["U_in"], contoh.X_CONTOH
    v11 = (U[:, 0] ** 2 * X[:, 0]).sum() / (U[:, 0] ** 2).sum()
    assert r["V"][0, 0] == pytest.approx(v11)
    assert "0.7688" in contoh.teks_pusat(X, U, 2.0, 0, 0)          # angka disubstitusi, bukan teks tetap


def test_contoh_cocok_membership_demo():
    r = contoh.langkah_fcm(contoh.X_CONTOH, contoh.U0_CONTOH, m=2.5, n_iter=2)[-1]
    u = demo_data.membership(contoh.X_CONTOH[0], r["V"], m=2.5)
    assert np.allclose(u, r["U_out"][0])


def test_contoh_knn_rumus_dan_alpha():
    X, U = contoh.X_CONTOH, contoh.langkah_fcm(contoh.X_CONTOH, contoh.U0_CONTOH, 2.0, 2)[-1]["U_out"]
    r = contoh.jarak_knn_contoh(X, U, 0, 1, 0.5)
    assert r["d"] == pytest.approx(0.5 * np.linalg.norm(X[0] - X[1]) + 0.5 * np.linalg.norm(U[0] - U[1]))
    assert r["skor"] == pytest.approx(1 / (1 + r["d"]))
    assert contoh.jarak_knn_contoh(X, U, 0, 1, 1.0)["d"] == pytest.approx(np.linalg.norm(X[0] - X[1]))


def test_rincian_jarak_sama_dengan_recommend():
    raw = demo_data.buat_data_mentah(n=300)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.csv"); raw.to_csv(p, index=False)
        X, f, _ = demo_data.build_features(demo_data.load_and_clean(p))
    U = demo_data.fcm(X, 4)["U"]
    reco = demo_data.recommend(5, X, U, k=8, alpha=0.3)
    rin = interpret.rincian_jarak(X, U, X[5], U[5], reco["idx"], 0.3)
    assert np.allclose(rin["d"], reco["distance"])
    assert np.allclose(rin["bagian_fitur"] + rin["bagian_mem"], rin["d"])


def test_prediksi_dan_galat():
    assert interpret.prediksi_rating([6, 8], [1, 1]) == pytest.approx(7)
    assert interpret.prediksi_rating([6, 8], [3, 1]) == pytest.approx(6.5)
    rmse, mae = interpret.galat_prediksi([5, 7], [6, 5])
    assert rmse == pytest.approx(np.sqrt((1 + 4) / 2)) and mae == pytest.approx(1.5)


def test_grafik_baru():
    rng = np.random.default_rng(1)
    U = rng.dirichlet(np.ones(4), size=30)
    rin = pd.DataFrame({"bagian_fitur": rng.random(5), "bagian_mem": rng.random(5), "d": rng.random(5)})
    figs = [
        plots.plot_distance_breakdown(rin, list("abcde"), 0.5),
        plots.plot_membership_compare(U[:6], [f"F{i}" for i in range(6)]),
        plots.plot_rating_comparison(7.1, list("abc"), [6, 7, 8]),
        plots.plot_actual_vs_pred([5, 6, 7], [5.5, 6, 6.8], [5.2, 6.4, 7.1], ["a", "b", "c"]),
        plots.plot_alpha_sweep(pd.DataFrame({"alpha": [0, .5, 1], "precision": [.8, .9, .85]})),
        plots.plot_center_trajectory(rng.normal(size=(30, 2)), [rng.normal(size=(4, 2)) for _ in range(3)], ["a", "b", "c"], U),
        plots.plot_membership_function(np.linspace(0, 1, 20), rng.dirichlet(np.ones(4), size=20), "Rating"),
        plots.plot_contoh_fcm(contoh.X_CONTOH, contoh.NAMA_FILM, [contoh.U0_CONTOH[:2, :2]], ["a"], contoh.U0_CONTOH),
    ]
    assert all(len(f.data) > 0 for f in figs)
    assert "digraph" in plots.alur_algoritma_dot()


# ---- jalur input tidak valid: pesan harus menjelaskan masalah dan cara memperbaiki -------------

def test_error_film_manual():
    p = interpret.validasi_film_manual("", 11, 10, -1, 0, 1800, [])
    teks = " ".join(p)
    for kata in ("Judul film belum diisi", "Genre belum dipilih", "Rating", "Jumlah vote", "Popularitas", "Durasi", "Tahun rilis"):
        assert kata in teks
    assert interpret.validasi_film_manual("X", 7, 500, 10, 100, 2000, ["Drama"]) == []


def test_error_parameter():
    assert len(interpret.validasi_parameter(1, 5.0, 1.0, 10)) == 4
    assert interpret.validasi_parameter(5, 2.0, 1e-5, 300) == []


@pytest.mark.parametrize("exc,kata", [
    (FileNotFoundError("x"), "File tidak ditemukan"),
    (NotImplementedError(), "belum diimplementasikan"),
    (pd.errors.EmptyDataError("x"), "CSV kosong"),
    (pd.errors.ParserError("x"), "tidak bisa dibaca sebagai CSV"),
    (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "x"), "UTF-8"),
    (ValueError("jelek"), "Periksa kembali"),
])
def test_pesan_error_ramah(exc, kata):
    assert kata in interpret.pesan_error_ramah(exc)


def test_csv_tipe_data_salah():
    df = pd.DataFrame({k: [1, 2] for k in interpret.KOLOM_WAJIB})
    df["vote_average"] = ["abc", "7"]
    df["release_date"] = ["bukan tanggal", "x"]
    pesan = " ".join(interpret.validasi_csv_cadangan(df))
    assert "vote_average" in pesan and "bukan angka" in pesan and "release_date" in pesan
    assert "baris data" in interpret.validasi_csv_cadangan(pd.DataFrame(columns=interpret.KOLOM_WAJIB))[0]
