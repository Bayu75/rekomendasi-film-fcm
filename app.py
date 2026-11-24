"""FuzzyFlix · titik masuk aplikasi dan halaman Beranda.

Jalankan dengan:  streamlit run app.py

Anggota 4: GUI & Visualisasi. ``st.navigation`` dipakai agar halaman pertama
bernama "Beranda" (bukan "app") dan keenam halaman muncul berurutan di sidebar.
File halaman tetap berada di folder ``pages/`` sesuai struktur PRD bagian 9.
"""

from __future__ import annotations

import streamlit as st

from src import provider
from src.interpret import fmt_int
from src import komponen
from src import ui
from src.ui import HALAMAN, page_setup


def beranda() -> None:
    """Halaman Beranda (F-01, F-02): nama, deskripsi, tujuan, alur, status."""
    page_setup("FuzzyFlix", "Rekomendasi film berdasarkan kemiripan fuzzy, bukan hanya satu label cluster.", aktif="beranda")

    st.markdown(
        "**FuzzyFlix** membantu Anda menemukan film yang mirip dengan film yang sudah Anda sukai. "
        "Sistem mengelompokkan ribuan film dengan **Fuzzy C-Means**, sehingga setiap film punya "
        "**derajat keanggotaan** ke beberapa cluster. Derajat keanggotaan itu kemudian masuk ke "
        "rumus jarak **KNN** untuk mencari film yang paling mirip."
    )

    st.subheader("Tujuan sistem")
    st.markdown(
        "- Memberi rekomendasi film **beserta penjelasan proses fuzzy-nya**.\n"
        "- Membuktikan secara terukur kontribusi FCM, dengan FPC, silhouette, dan Precision@K "
        "dibanding KNN biasa."
    )

    st.subheader("Kombinasi metode")
    st.markdown(
        "**Fuzzy C-Means + KNN, terintegrasi:** hasil FCM bukan label yang dibuang, melainkan vektor derajat keanggotaan $u$ "
        "yang masuk ke rumus jarak KNN:"
    )
    st.latex(r"d(q,i) = \alpha\,\lVert x_q - x_i\rVert + (1-\alpha)\,\lVert u_q - u_i\rVert")
    st.caption("Penjelasan lengkap, contoh perhitungan, dan daftar library ada di halaman Informasi.")

    st.divider()
    st.subheader("Alur singkat")
    k1, k2, k3 = st.columns(3)
    kartu = [
        (k1, "1", "Input", "Pilih dataset bawaan, unggah CSV, atau masukkan ciri film baru.", "database"),
        (k2, "2", "Proses", "Atur parameter FCM, jalankan clustering, dan lihat konvergensi J_m.", "gear"),
        (k3, "3", "Hasil", "Pilih film acuan, dapatkan top-k rekomendasi, dan baca interpretasinya.", "star"),
    ]
    for kolom, nomor, judul, isi, nama_ikon in kartu:
        with kolom:
            st.html(komponen.kartu_langkah(nomor, judul, isi, nama_ikon))
    st.caption("Setelah itu, halaman Evaluasi mengukur kualitas cluster dan rekomendasi, "
               "dan halaman Informasi menjelaskan metode serta cara pemakaian.")

    st.divider()
    if st.button("Mulai", type="primary", help="Buka halaman Input Data untuk memilih dataset."):
        st.switch_page(HALAMAN["input"])

    st.subheader("Status saat ini")
    a, b = st.columns(2)
    ss = st.session_state
    with a:
        if provider.data_siap():
            st.success(f"Data siap dipakai ({fmt_int(len(ss.df_film))} film). Sumber: {ss.data_sumber}.")
        else:
            st.warning("Data belum dimuat. Buka halaman Input Data.")
    with b:
        if provider.model_siap():
            st.success(f"Model FCM siap (c = {ss.fcm_result['V'].shape[0]}).")
        else:
            st.warning("Model belum dijalankan. Buka halaman Proses lalu klik Jalankan Clustering.")


st.set_page_config(page_title="FuzzyFlix", page_icon="🎬", layout="wide")

ui.PAGES.update({
    "beranda": st.Page(beranda, title="Beranda", icon="🏠", default=True),
    "input": st.Page(HALAMAN["input"], title="Input Data", icon="📥", url_path="input-data"),
    "proses": st.Page(HALAMAN["proses"], title="Proses", icon="⚙️", url_path="proses"),
    "hasil": st.Page(HALAMAN["hasil"], title="Hasil Rekomendasi", icon="🎯", url_path="hasil"),
    "evaluasi": st.Page(HALAMAN["evaluasi"], title="Evaluasi", icon="📊", url_path="evaluasi"),
    "informasi": st.Page(HALAMAN["informasi"], title="Informasi", icon="ℹ️", url_path="informasi"),
})
# Menu bawaan disembunyikan; menu tampil dari ui.sidebar_status() (logo, ikon, kartu Status Sistem).
navigasi = st.navigation(list(ui.PAGES.values()), position="hidden")
navigasi.run()
