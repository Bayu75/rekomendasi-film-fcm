"""Komponen GUI bersama: konfigurasi halaman, sidebar, banner, dan penjaga status.

Anggota 4: GUI & Visualisasi. Dipakai di setiap halaman supaya tampilan konsisten
(design-system.md bagian 4 dan 10).
"""

from __future__ import annotations

import streamlit as st
from streamlit.errors import StreamlitAPIException

from src import komponen, provider, tema
from src.interpret import fmt_int

HALAMAN = {
    "beranda": "app.py",
    "input": "pages/1_Input_Data.py",
    "proses": "pages/2_Proses.py",
    "hasil": "pages/3_Hasil.py",
    "evaluasi": "pages/4_Evaluasi.py",
    "informasi": "pages/5_Informasi.py",
}

# Objek st.Page yang didaftarkan app.py (dipakai st.page_link); kosong saat halaman dites sendiri.
PAGES: dict = {}

# (kunci halaman, label menu, ikon material)
NAVIGASI = [
    ("beranda", "Beranda", ":material/home:"),
    ("input", "Input Data", ":material/database:"),
    ("proses", "Proses", ":material/settings:"),
    ("hasil", "Hasil Rekomendasi", ":material/star:"),
    ("evaluasi", "Evaluasi", ":material/bar_chart:"),
    ("informasi", "Informasi", ":material/info:"),
]


def page_setup(judul: str, keterangan: str, nama_ikon: str = "clapper", aktif: str | None = None) -> None:
    """Awal setiap halaman: set_page_config, tema, sesi, sidebar, header, banner demo.

    ``st.set_page_config`` sama di semua halaman (checklist design-system 10).
    Bila entrypoint ``app.py`` sudah memanggilnya, panggilan ulang diabaikan.
    """
    try:
        st.set_page_config(page_title="FuzzyFlix", page_icon="🎬", layout="wide")
    except StreamlitAPIException:
        pass
    st.html(tema.CSS + tema.css_menu_aktif(aktif))
    provider.init_session()
    sidebar_status()
    st.html(komponen.chip_pengguna("FuzzyFlix"))
    st.html(komponen.header_halaman(judul, keterangan, nama_ikon))
    banner_demo()


def _tautan(kunci: str, label: str, ikon_material: str) -> None:
    """Satu item menu sidebar; memakai objek st.Page bila ada, jika tidak jalur file."""
    with st.container(key=f"nav_{kunci}"):
        try:
            st.page_link(PAGES.get(kunci) or HALAMAN[kunci], label=label, icon=ikon_material, width="stretch")
        except StreamlitAPIException:  # halaman dijalankan sendiri (mis. saat pengujian), tanpa st.navigation
            st.caption(label)


def sidebar_status() -> None:
    """Sidebar: logo, menu, kartu Status Sistem, dan tagline (design-system 4.1)."""
    ss = st.session_state
    with st.sidebar:
        st.html(komponen.logo())
        for kunci, label, ikon_material in NAVIGASI:
            _tautan(kunci, label, ikon_material)
        baris = []
        if provider.data_siap():
            baris.append((f"Data dimuat ({fmt_int(len(ss.df_film))} film)", "ok"))
        else:
            baris.append(("Data belum dimuat", "peringatan"))
        if provider.model_siap():
            baris.append((f"Model FCM (c = {ss.fcm_result['V'].shape[0]})", "cek"))
        else:
            baris.append(("Model belum dijalankan", "peringatan"))
        if provider.dari_demo():
            baris.append(("Mode demo", "peringatan"))
        st.html(komponen.kartu_status(baris))
        st.html('<div class="ff-tagline">Temukan film favoritmu berikutnya ✨</div>')


def banner_demo() -> None:
    """Banner bahwa data/algoritma demo dipakai (kosong jika semuanya asli)."""
    baris = provider.info_demo()
    if baris:
        item = "".join(f"<li>{komponen.md_mini(b)}</li>" for b in baris)
        st.html(f'<div class="ff-demo"><b>🧪 Mode demo.</b> Angka dan film di halaman ini bukan hasil sebenarnya.'
                f'<ul>{item}</ul>Saat modul dan <code>models/</code> asli tersedia, aplikasi otomatis memakainya '
                f'(lihat <code>docs/GUI-notes.md</code>).</div>')


def butuh_data() -> bool:
    """Jika data belum ada: tampilkan peringatan + tombol ke Input Data, return False."""
    if provider.data_siap():
        return True
    st.warning("Data belum dimuat. Buka halaman Input Data lalu pilih dataset.")
    st.page_link(HALAMAN["input"], label="Buka Input Data", icon="📥")
    return False


def butuh_model() -> bool:
    """Jika model belum ada: peringatan + tombol ke Proses (design-system bagian 9)."""
    if not butuh_data():
        return False
    if provider.model_siap():
        return True
    st.warning("Model belum dijalankan. Buka halaman Proses lalu klik Jalankan Clustering.")
    st.page_link(HALAMAN["proses"], label="Buka Proses", icon="⚙️")
    return False


def tampilkan_error(exc: BaseException, konteks: str = "") -> None:
    """``st.error`` ramah (apa yang salah + cara memperbaiki); detail teknis di expander."""
    from src.interpret import pesan_error_ramah
    st.error(pesan_error_ramah(exc, konteks))
    with st.expander("Detail teknis (untuk pengembang)"):
        st.code(f"{type(exc).__name__}: {exc}")


def formula_expander(judul: str = "Bagaimana ini dihitung?"):
    """Expander standar rumus (checklist design-system 10)."""
    return st.expander(judul)
