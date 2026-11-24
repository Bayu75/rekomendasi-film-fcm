"""Uji asap GUI: setiap halaman dirender dengan data demo tanpa exception.

Jalankan:  pytest tests/test_gui_smoke.py
Memakai ``streamlit.testing.v1.AppTest`` (tanpa browser).
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
HALAMAN = ["app.py", "pages/1_Input_Data.py", "pages/2_Proses.py", "pages/3_Hasil.py",
           "pages/4_Evaluasi.py", "pages/5_Informasi.py"]


@pytest.mark.parametrize("berkas", HALAMAN)
def test_halaman_render(berkas):
    at = AppTest.from_file(str(ROOT / berkas), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    # hanya satu tombol utama per halaman (design-system bagian 10)
    primer = [b for b in at.button if b.proto.type == "primary"]
    assert len(primer) <= 1


def test_hasil_cari_rekomendasi():
    at = AppTest.from_file(str(ROOT / "pages/3_Hasil.py"), default_timeout=180).run()
    next(b for b in at.button if b.label == "Cari Rekomendasi").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error


def test_proses_jalankan_clustering():
    at = AppTest.from_file(str(ROOT / "pages/2_Proses.py"), default_timeout=180).run()
    next(b for b in at.button if b.label == "Jalankan Clustering").click()
    at.run()
    assert not at.exception and not at.error
    assert any("Clustering selesai" in s.value for s in at.success)


def test_evaluasi_jalankan_pengujian():
    at = AppTest.from_file(str(ROOT / "pages/4_Evaluasi.py"), default_timeout=180).run()
    next(b for b in at.button if b.label == "Jalankan Pengujian").click()
    at.run()
    assert not at.exception and not at.error


def test_input_manual_validasi_dan_film_baru():
    at = AppTest.from_file(str(ROOT / "pages/1_Input_Data.py"), default_timeout=180).run()
    # tanpa genre -> pesan error yang jelas
    next(b for b in at.button if b.label == "Simpan Film Baru").click()
    at.run()
    assert any("Genre belum dipilih" in e.value for e in at.error)
    # dengan genre -> film baru tersimpan
    at.multiselect[0].select("Drama")
    next(b for b in at.button if b.label == "Simpan Film Baru").click()
    at.run()
    assert not at.exception
    assert at.session_state["new_film"] is not None


def test_hasil_dengan_film_baru():
    at = AppTest.from_file(str(ROOT / "pages/1_Input_Data.py"), default_timeout=180).run()
    at.multiselect[0].select("Action")
    next(b for b in at.button if b.label == "Simpan Film Baru").click()
    at.run()
    at2 = AppTest.from_file(str(ROOT / "pages/3_Hasil.py"), default_timeout=180).run()
    at2.session_state["new_film"] = at.session_state["new_film"]  # sesi baru, salin film baru
    at2.run()
    # film baru muncul sebagai opsi pertama
    assert at2.selectbox[0].options[0].startswith("➕")
    next(b for b in at2.button if b.label == "Cari Rekomendasi").click()
    at2.run()
    assert not at2.exception and not at2.error
    assert at2.session_state["last_reco"]["q_idx"] is None


def test_validasi_csv_pesan_ramah():
    import pandas as pd
    from src.interpret import validasi_csv_cadangan
    pesan = validasi_csv_cadangan(pd.DataFrame({"id": [1], "title": ["x"]}))
    assert pesan and "runtime" in pesan[0] and "Pastikan CSV memiliki kolom" in pesan[0]


def test_proses_punya_hasil_antara_dan_fungsi_keanggotaan():
    at = AppTest.from_file(str(ROOT / "pages/2_Proses.py"), default_timeout=180).run()
    assert not at.exception
    snap = at.session_state["hasil_antara"]
    assert snap and snap[-1]["label"].startswith("Akhir")
    judul = " ".join(s.value for s in at.subheader)
    for kata in ("Fungsi keanggotaan FCM", "Hasil antara", "Parameter yang digunakan"):
        assert kata in judul


def test_hasil_ada_uraian_integrasi():
    at = AppTest.from_file(str(ROOT / "pages/3_Hasil.py"), default_timeout=180).run()
    next(b for b in at.button if b.label == "Cari Rekomendasi").click()
    at.run()
    assert not at.exception and not at.warning
    assert any("Integrasi FCM" in s.value for s in at.subheader)
    reco = at.session_state["last_reco"]
    assert reco["selisih_jarak"] < 1e-6 and len(reco["idx_base"]) == len(reco["idx"])


def test_evaluasi_ada_rmse_aktual_vs_prediksi_dan_sweep():
    at = AppTest.from_file(str(ROOT / "pages/4_Evaluasi.py"), default_timeout=180).run()
    next(b for b in at.button if b.label == "Jalankan Pengujian").click()
    at.run()
    assert not at.exception and not at.error
    ev = at.session_state["eval_hasil"]
    assert ev["rmse_fuzzy"] > 0 and len(ev["sweep"]) == 5 and len(ev["aktual"]) == 200
    assert any("aktual vs prediksi" in s.value for s in at.subheader)


def test_informasi_semua_tab_dan_contoh():
    at = AppTest.from_file(str(ROOT / "pages/5_Informasi.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5
    # slider contoh perhitungan dapat diubah tanpa error
    at.slider(key="contoh_m").set_value(1.5)
    at.slider(key="contoh_iter").set_value(3)
    at.run()
    assert not at.exception and not at.error
