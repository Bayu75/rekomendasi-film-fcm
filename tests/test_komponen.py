"""Uji komponen tampilan Cinematic Dark: escape HTML, poster pengganti tanpa jaringan, metrik, tabel, halaman Hasil."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src import komponen, plots, tema

ROOT = Path(__file__).resolve().parents[1]


def test_escape_judul_dan_genre():
    jahat = '<script>alert(1)</script> & "x"'
    html = komponen.kartu_film_pilihan(jahat, 2001, "7.5", "100 menit", [jahat], jahat, None, [("K", "10")])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_tabel_rekomendasi_escape_dan_bar():
    html = komponen.tabel_rekomendasi([{"no": 1, "judul": "<b>Film</b>", "tahun": 2000, "genre": "A&B",
                                        "skor": 0.5, "poster_path": None}])
    assert "<b>Film</b>" not in html and "&lt;b&gt;Film&lt;/b&gt;" in html
    assert "width:50.0%" in html and "0.500" in html


def test_poster_pengganti_tanpa_jaringan():
    # tanpa poster_path: tidak ada <img>, hanya inisial judul
    html = komponen.poster("Langit Biru", None, 100)
    assert "ff-poster-img" not in html and ">LB<" in html
    for kosong in ("", "nan", None, float("nan")):
        assert komponen.url_poster(kosong) is None


def test_url_poster_hanya_jalur_tmdb_valid():
    assert komponen.url_poster("/abc123.jpg") == "https://image.tmdb.org/t/p/w185/abc123.jpg"
    assert komponen.url_poster("http://jahat.example/x.jpg") is None
    assert komponen.url_poster('/x" onerror="alert(1)') is None
    assert "ff-poster-img" in komponen.poster("X", "/abc.jpg", 100)


def test_inisial_dan_judul_berwarna():
    assert komponen.inisial("the dark knight") == "TD"
    assert komponen.inisial("") == "?"
    assert komponen.judul_berwarna("Hasil Rekomendasi") == ("Hasil", "Rekomendasi")
    assert komponen.judul_berwarna("FuzzyFlix") == ("Fuzzy", "Flix")


def test_kartu_metrik_dan_sparkline_tanpa_data():
    html = komponen.kartu_metrik("target", komponen.AMBER, "-", "Precision@10", None)
    assert "Precision@10" in html and "ff-spark" in html
    assert "background-image:url('data:image/svg+xml" in komponen.sparkline([1, 3, 2], komponen.MERAH)
    # nilai tak hingga/NaN tidak merusak
    assert "ff-spark" in komponen.sparkline([float("nan"), 1.0, float("inf")], komponen.MERAH)


def test_callout_md_mini_aman():
    html = komponen.callout_ringkasan("**Film** <i>x</i> `k`")
    assert "<b>Film</b>" in html and "<i>x</i>" not in html and "<code>k</code>" in html


def test_tema_menu_aktif_aman():
    assert "st-key-nav_hasil" in tema.css_menu_aktif("hasil")
    assert tema.css_menu_aktif('x"}</style><script>') == ""
    assert tema.css_menu_aktif(None) == ""


def test_dot_tanpa_penanda_tersisa():
    for dot in (plots.arsitektur_dot(), plots.alur_algoritma_dot()):
        assert "__" not in dot and "digraph" in dot


def test_hasil_halaman_dengan_tema_baru():
    at = AppTest.from_file(str(ROOT / "pages/3_Hasil.py"), default_timeout=180).run()
    next(b for b in at.button if b.label == "Cari Rekomendasi").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    reco = at.session_state["last_reco"]
    assert reco is not None and len(reco["tabel"]) == 10
    # poster/sinopsis opsional tersedia di film_view (demo: kosong) tanpa jaringan
    fv = at.session_state["film_view"]
    assert "poster_path" in fv.columns and "overview" in fv.columns


def test_metrik_ringkas_demo():
    src = '''
import streamlit as st
from src import provider
provider.init_session()
st.session_state["_mt"] = provider.metrik_ringkas()
'''
    tmp = ROOT / "tests" / "_tmp_metrik_app.py"
    tmp.write_text(src)
    try:
        at2 = AppTest.from_file(str(tmp), default_timeout=180).run()
        mt = at2.session_state["_mt"]
    finally:
        tmp.unlink(missing_ok=True)
    assert 0 < mt["fpc"] <= 1
    assert mt["silhouette"] is not None and -1 <= mt["silhouette"] <= 1
    assert mt["precision10"] is not None and 0 <= mt["precision10"] <= 1
    assert len(mt["seri_prec"]) == 3


def test_ekstra_film_dari_csv(tmp_path):
    import pandas as pd
    from src import provider
    csv = tmp_path / "m.csv"
    pd.DataFrame({"id": [1, 2], "title": ["A", "B"], "poster_path": ["/a.jpg", None], "overview": ["Cerita A", "Cerita B"]}).to_csv(csv, index=False)
    e = provider._ekstra_film(str(csv))
    assert e is not None and set(["id", "poster_path", "overview", "_kunci"]) <= set(e.columns)
    assert e.set_index("_kunci").loc["1", "overview"] == "Cerita A"
    # tanpa kolom opsional -> None (GUI memakai poster pengganti)
    pd.DataFrame({"id": [1], "title": ["A"]}).to_csv(csv, index=False)
    assert provider._ekstra_film(str(csv)) is None
