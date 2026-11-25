"""Halaman Input Data (F-03 s.d. F-07). Anggota 4: GUI & Visualisasi."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import provider
from src.interpret import (KOLOM_WAJIB, PENJELASAN_VARIABEL, RENTANG, fmt_int,
                           validasi_film_manual)
from src.plots import plot_query_membership
from src.ui import HALAMAN, page_setup, tampilkan_error

page_setup("Input Data", "Pilih sumber data, unggah CSV, atau masukkan ciri film baru secara manual.", aktif="input")
ss = st.session_state

# ---------------------------------------------------------------------------
# Bagian 1: sumber data (F-03, F-04)
# ---------------------------------------------------------------------------
st.subheader("Sumber data")
sumber = st.radio(
    "Pilih sumber dataset", ["Dataset bawaan", "Unggah CSV"], horizontal=True,
    help="Dataset bawaan = data/raw/movies_metadata.csv (The Movies Dataset). "
         "Unggahan harus memiliki kolom wajib yang sama.")

if sumber == "Dataset bawaan":
    if provider._muat_artefak_model() is not None:
        st.markdown("Data asli tim ditemukan (`data/processed/` dan `models/U.npy`, `V.npy`, `pca_fcm.joblib`). "
                    "Klik tombol untuk memuatnya.")
    elif provider.CSV_BAWAAN.exists():
        st.markdown("File `data/raw/movies_metadata.csv` ditemukan. Klik tombol untuk membersihkan data dan membentuk fitur.")
    else:
        st.warning("File `data/raw/movies_metadata.csv` belum ada. Unduh The Movies Dataset dari Kaggle dan taruh "
                   "di folder `data/raw/`. Selama belum ada, tombol di bawah memuat **data demo**.")
    if st.button("Pakai Dataset Bawaan", type="primary",
                 help="Menjalankan pembersihan (filter vote_count ≥ 50, runtime > 0) dan pembentukan fitur."):
        try:
            with st.spinner("Membersihkan data dan membentuk fitur…"):
                hasil = provider.muat_dataset_bawaan()
            if hasil == "demo-fallback":
                st.info("Dataset bawaan belum ada di komputer ini, sehingga **data demo** dimuat.")
            st.success(f"Dataset dimuat: {fmt_int(len(ss.df_film))} film. "
                       "Lanjut ke halaman Proses untuk menjalankan clustering.")
        except Exception as e:  # noqa: BLE001
            tampilkan_error(e, "Dataset bawaan tidak bisa dimuat.")
else:
    st.markdown("Kolom wajib: " + ", ".join(f"`{k}`" for k in KOLOM_WAJIB) + ".")
    berkas = st.file_uploader(
        "Unggah dataset film (CSV)", type="csv",
        help="Ukuran maksimal 200 MB. Kolom wajib: " + ", ".join(KOLOM_WAJIB) + ".")
    if berkas is not None:
        df_up, galat = None, None
        try:
            df_up = pd.read_csv(berkas, low_memory=False)
        except Exception as e:  # noqa: BLE001
            galat = e
        if galat is not None:
            tampilkan_error(galat, "File tidak bisa dibaca.")
        else:
            pesan = provider.validasi_csv(df_up)
            if pesan:
                st.error("CSV belum bisa dipakai. Perbaiki hal berikut lalu unggah ulang:")
                for p in pesan:
                    st.error(p)
            else:
                st.success(f"CSV valid: {fmt_int(len(df_up))} baris dan semua kolom wajib ditemukan.")
                st.dataframe(df_up.head(5), hide_index=True, width="stretch")
                if st.button("Unggah Dataset", type="primary",
                             help="Membersihkan data unggahan dan membentuk fitur untuk clustering."):
                    try:
                        berkas.seek(0)
                        with st.spinner("Membersihkan data dan membentuk fitur…"):
                            provider.muat_dataset_unggahan(berkas.getvalue(), berkas.name)
                        st.success(f"Dataset unggahan dimuat: {fmt_int(len(ss.df_film))} film setelah pembersihan. "
                                   "Lanjut ke halaman Proses.")
                    except Exception as e:  # noqa: BLE001
                        tampilkan_error(e, "Dataset unggahan tidak bisa diproses.")

if ss.get("data_mode") != "demo" or ss.get("data_sumber", "").startswith("unggahan"):
    if st.button("Kembali ke Data Demo", help="Memuat ulang data demo sintetis beserta model demo."):
        provider.gunakan_data_demo()
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Bagian 2: input manual film baru (F-05, F-07)
# ---------------------------------------------------------------------------
st.subheader("Input film baru secara manual")
st.markdown("Masukkan ciri film yang belum ada di dataset. Sistem menghitung derajat keanggotaannya dari "
            "pusat cluster yang tersimpan, lalu film ini bisa dipilih sebagai film acuan di halaman Hasil.")

if not provider.model_siap():
    st.warning("Model belum dijalankan. Buka halaman Proses lalu klik Jalankan Clustering "
               "sebelum memasukkan film baru.")
    st.page_link(HALAMAN["proses"], label="Buka Proses", icon="⚙️")
else:
    with st.form("form_film_baru"):
        judul = st.text_input("Judul film baru", value="Film Baru Saya",
                              help="Hanya untuk tampilan, tidak dipakai dalam perhitungan.")
        k1, k2, k3 = st.columns(3)
        rating = k1.number_input("Rating (0–10)", 0.0, 10.0, 7.0, 0.1, format="%.1f",
                                 help="Rata-rata nilai penonton (vote_average).")
        vote = k2.number_input("Jumlah vote", RENTANG["vote_count"][0], RENTANG["vote_count"][1], 500, 10,
                               help="Banyaknya penonton yang memberi nilai (vote_count); minimal 50 seperti filter dataset.")
        pop = k3.number_input("Popularitas", 0.0, float(RENTANG["popularity"][1]), 15.0, 0.5, format="%.1f",
                              help="Skor popularitas TMDB. Film populer biasanya 10–50.")
        k4, k5 = st.columns(2)
        durasi = k4.number_input("Durasi (menit)", RENTANG["runtime"][0], RENTANG["runtime"][1], 110, 5,
                                 help="Lama film dalam menit.")
        tahun = k5.number_input("Tahun rilis", RENTANG["year"][0], RENTANG["year"][1], 2015, 1,
                                help="Tahun film dirilis.")
        genres = st.multiselect("Genre", provider.daftar_genre(), default=[], placeholder="Pilih satu atau lebih genre",
                                help="Pilih minimal satu genre.")
        kirim = st.form_submit_button("Simpan Film Baru")
    if kirim:
        galat = validasi_film_manual(judul, rating, vote, pop, durasi, tahun, genres)
        if galat:
            for g in galat:
                st.error(g)
        else:
            try:
                provider.simpan_film_baru(judul, rating, vote, pop, durasi, tahun, genres)
                st.success(f"Film baru **{judul}** disimpan. Buka halaman Hasil dan pilih film ini sebagai film acuan.")
            except Exception as e:  # noqa: BLE001
                tampilkan_error(e, "Film baru tidak bisa diproses.")
    fb = ss.get("new_film")
    if fb:
        st.markdown(f"**Film baru tersimpan:** {fb['title']}")
        st.plotly_chart(plot_query_membership(fb["u"], ss.nama_cluster,
                                              title=f"Derajat keanggotaan: {fb['title']}"),
                        width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Bagian 3: penjelasan variabel (F-06) dan preview data
# ---------------------------------------------------------------------------
st.subheader("Penjelasan variabel")
st.dataframe(PENJELASAN_VARIABEL, hide_index=True, width="stretch")

st.subheader("Preview data (10 baris)")
if provider.data_siap():
    fv = ss.film_view
    prev = pd.DataFrame({
        "Judul": fv["title"].astype(str), "Tahun": fv["_year"].round().astype("Int64"),
        "Genre": fv["_genres_text"], "Rating": pd.to_numeric(fv["vote_average"], errors="coerce").round(1),
        "Jumlah vote": fv.get("vote_count"), "Popularitas": pd.to_numeric(fv.get("popularity"), errors="coerce").round(1),
        "Durasi": fv.get("runtime")}).head(10)
    m1, m2, m3 = st.columns(3)
    m1.metric("Jumlah film", fmt_int(len(fv)))
    m2.metric("Jumlah fitur (kolom X)", ss.X.shape[1])
    m3.metric("Rentang tahun", f"{int(fv['_year'].min())}–{int(fv['_year'].max())}")
    st.dataframe(prev, hide_index=True, width="stretch")
else:
    st.info("Belum ada data untuk dipratinjau.")
