"""Halaman Informasi Sistem (F-23, F-24; brief E6, F, G). Anggota 4: GUI & Visualisasi."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import contoh, provider
from src.interpret import PENJELASAN_VARIABEL
from src.plots import alur_algoritma_dot, arsitektur_dot, plot_contoh_fcm
from src.ui import page_setup, tampilkan_error

page_setup("Informasi Sistem", "Metode, variabel, parameter, algoritma, library, contoh perhitungan, dan petunjuk penggunaan.", aktif="informasi")

tab_metode, tab_lib, tab_contoh, tab_arsitek, tab_petunjuk = st.tabs(
    ["Metode & Integrasi", "Library & Kode", "Contoh Perhitungan", "Arsitektur & Alur", "Petunjuk & Anggota"])

# ===========================================================================
# Tab 1: metode, integrasi, kartu input-parameter-rumus-proses-output
# ===========================================================================
with tab_metode:
    st.subheader("Kombinasi metode: Fuzzy C-Means + KNN (terintegrasi)")
    st.markdown(
        "- **Mengapa FCM?** Film berciri campuran (mis. aksi + komedi) tidak cocok dipaksa ke satu kelompok. "
        "FCM memberi **derajat keanggotaan** 0–1 ke setiap cluster sehingga ciri campuran tetap tercatat.\n"
        "- **Mengapa KNN?** Rekomendasi pada dasarnya mencari film paling mirip, dan KNN melakukannya langsung lewat jarak.\n"
        "- **Integrasi:** vektor keanggotaan FCM masuk ke **dalam rumus jarak KNN**. Keduanya tidak dijalankan terpisah. "
        "Hasilnya terlihat di halaman Hasil (uraian jarak dan perbandingan dengan KNN biasa)."
    )
    st.latex(r"d(q,i) = \alpha\,\lVert x_q - x_i\rVert + (1-\alpha)\,\lVert u_q - u_i\rVert")
    st.caption("FCM menghasilkan $u$; KNN memakainya. Dengan α = 1 sistem menjadi KNN biasa (pembanding).")

    st.divider()
    st.subheader("Kartu metode: input, parameter, rumus, proses, keluaran")

    def kartu(judul, input_, parameter, rumus, proses, keluaran):
        with st.container(border=True):
            st.markdown(f"#### {judul}")
            k1, k2 = st.columns(2)
            k1.markdown(f"**Input**\n\n{input_}")
            k2.markdown(f"**Parameter**\n\n{parameter}")
            st.markdown("**Rumus utama**")
            for r in rumus:
                st.latex(r)
            k3, k4 = st.columns(2)
            k3.markdown(f"**Proses iterasi / pembelajaran**\n\n{proses}")
            k4.markdown(f"**Keluaran**\n\n{keluaran}")

    kartu("1. Fuzzy C-Means (FCM)",
          "Matriks fitur ternormalisasi $X$ (n film × f fitur).",
          "c (jumlah cluster 2–10), m (fuzzifier 1,1–3,0; default 2), ε (1e-6–1e-3), maxiter (50–1000), seed 42.",
          [r"v_j = \frac{\sum_i u_{ij}^{m} x_i}{\sum_i u_{ij}^{m}}",
           r"u_{ij} = \frac{1}{\sum_{k=1}^{c}\left(d_{ij}/d_{ik}\right)^{2/(m-1)}}",
           r"J_m = \sum_i\sum_j u_{ij}^{m}\, d_{ij}^{2}"],
          "1) inisialisasi; 2) hitung pusat $V$; 3) perbarui $U$; 4) hitung $J_m$ dan $\\max|U_{baru}-U_{lama}|$; "
          "ulangi 2–4 sampai $< ε$ atau mencapai maxiter. $J_m$ selalu turun sampai konvergen.",
          "$V$ (c × f), $U$ (n × c), riwayat $J_m$, jumlah iterasi, FPC.")
    kartu("2. KNN berbobot keanggotaan fuzzy (integrasi)",
          "$X$, $U$ dari FCM, film acuan $q$ (dari dataset, atau film baru dengan $u_q$ dari rumus keanggotaan memakai $V$).",
          "k (5–20, default 10), α (0–1, default 0,5).",
          [r"d(q,i) = \alpha\lVert x_q-x_i\rVert + (1-\alpha)\lVert u_q-u_i\rVert", r"\text{skor kemiripan} = \frac{1}{1+d(q,i)}"],
          "Tidak ada pembelajaran iteratif: hitung jarak $q$ ke semua film, buang film acuan, urutkan, ambil $k$ terkecil.",
          "Tabel top-k (indeks, jarak, skor) lalu diperkaya judul, genre, cluster dominan.")
    kartu("3. PCA genre (pendukung)",
          "One-hot 20 genre tiap film.",
          "Jumlah komponen (±80% variansi).",
          [r"Z = (G - \bar G)\,W_k"],
          "Mencari arah variansi terbesar lalu memproyeksikan genre ke beberapa komponen.",
          "Komponen genre yang digabung ke $X$, dan PCA 2D untuk gambar peta cluster.")
    kartu("4. Evaluasi",
          "$X$, $U$, tabel film, 200 film acuan acak.",
          "K = 5/10/20, α, seed 42.",
          [r"\mathrm{FPC}=\frac1n\sum_i\sum_j u_{ij}^2", r"s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}",
           r"\mathrm{Precision@K}=\frac{\#\text{relevan}}{K}"],
          "Untuk tiap film acuan, ambil K rekomendasi (fuzzy dan KNN biasa), hitung relevan (≥1 genre sama), rata-ratakan.",
          "Tabel metrik, grafik Precision@K, RMSE/MAE, kualitas cluster.")

    st.divider()
    st.subheader("Variabel masukan")
    st.dataframe(PENJELASAN_VARIABEL, hide_index=True, width="stretch")
    st.subheader("Variabel keluaran")
    st.dataframe(pd.DataFrame([
        ("V", "Pusat cluster (c × f)", "Ciri rata-rata berbobot tiap cluster."),
        ("U", "Matriks derajat keanggotaan (n × c)", "Tiap baris berjumlah 1; ditampilkan sebagai heatmap dan bar."),
        ("J_m", "Fungsi objektif per iterasi", "Grafik konvergensi."),
        ("Skor kemiripan", "1 / (1 + d)", "Urutan rekomendasi, 0–1."),
        ("Cluster dominan", "argmax_j u_ij", "Dipakai untuk warna titik, silhouette, dan nama cluster."),
    ], columns=["Keluaran", "Bentuk", "Keterangan"]), hide_index=True, width="stretch")

    st.subheader("Parameter")
    st.dataframe(pd.DataFrame([
        ("c (jumlah cluster)", "dipilih otomatis (c = 3–8)", "2–10", "Banyaknya kelompok film."),
        ("m (fuzzifier)", "2,0", "1,1–3,0", "Makin besar, keanggotaan makin merata."),
        ("ε (toleransi)", "1e-5", "1e-6–1e-3", "Batas perubahan keanggotaan untuk berhenti."),
        ("Iterasi maksimum", "300", "50–1000", "Batas iterasi FCM."),
        ("k (rekomendasi)", "10", "5–20", "Jumlah film yang direkomendasikan."),
        ("α (bobot jarak fitur)", "0,5", "0,0–1,0", "α = 1 berarti KNN biasa."),
        ("seed", "42", "tetap", "Menjamin hasil bisa diulang."),
    ], columns=["Parameter", "Default", "Rentang di GUI", "Keterangan"]), hide_index=True, width="stretch")

# ===========================================================================
# Tab 2: library dan lokasi kode
# ===========================================================================
with tab_lib:
    st.subheader("Library yang dipakai")
    st.dataframe(pd.DataFrame([
        ("Streamlit", "Membangun GUI web: halaman, input, tombol, tabel, grafik.",
         "`st.navigation`, `st.button`, `st.slider`, `st.file_uploader`, `st.dataframe`, `st.plotly_chart`, `st.session_state`, `st.cache_data`", "app.py, pages/, src/ui.py"),
        ("NumPy", "Operasi matriks: jarak, pusat cluster, keanggotaan.", "`np.linalg.norm`, `@`, `np.argsort`, `np.argmax`", "fcm.py, knn_fuzzy.py, contoh.py"),
        ("pandas", "Membaca CSV, membersihkan dan menyajikan tabel film.", "`read_csv`, `to_numeric`, `DataFrame`, `merge`", "preprocessing.py, pages/"),
        ("scikit-learn", "Normalisasi, PCA, dan metrik cluster.", "`MinMaxScaler.fit_transform`, `PCA(n_components).fit_transform`, `silhouette_score`, `silhouette_samples`", "preprocessing.py, evaluation.py, provider.py"),
        ("scikit-fuzzy", "Pembanding implementasi FCM mandiri.",
         "`skfuzzy.cmeans(data, c, m, error, maxiter)` → pusat, U, jarak, riwayat J, FPC (data berbentuk fitur × sampel)", "fcm.py (uji pembanding)"),
        ("Plotly", "Grafik interaktif di GUI.", "`go.Scatter`, `go.Bar`, `go.Heatmap`, `go.Scatterpolar`, `make_subplots`", "src/plots.py"),
        ("Matplotlib", "Gambar statis untuk laporan.", "`plt.plot`, `plt.savefig`", "notebooks/"),
        ("joblib & pyarrow", "Menyimpan artefak dan tabel film.", "`joblib.dump/load`, `DataFrame.to_parquet`", "models/"),
        ("pytest", "Pengujian otomatis.", "`pytest.mark.parametrize`, `streamlit.testing.v1.AppTest`", "tests/"),
    ], columns=["Library", "Fungsi dalam proyek", "API utama yang dipakai", "Dipakai di"]).replace("`", "", regex=True),
        hide_index=True, width="stretch")

    with st.expander("Apa yang dihitung di dalam `skfuzzy.cmeans` dan scikit-learn?"):
        st.markdown(
            "- **`skfuzzy.cmeans`** menjalankan iterasi FCM yang sama dengan rumus di tab Metode: menghitung pusat dari $U^m$, "
            "jarak Euclidean, memperbarui $U$, lalu berhenti saat perubahan $U$ < `error` atau iterasi = `maxiter`. "
            "Keluarannya (`cntr, u, u0, d, jm, p, fpc`) dipakai sebagai pembanding hasil FCM mandiri.\n"
            "- **`MinMaxScaler`** menghitung $x' = (x - x_{min})/(x_{max} - x_{min})$ per fitur.\n"
            "- **`PCA`** menghitung vektor eigen matriks kovarians genre, lalu memproyeksikan data ke komponen utama.\n"
            "- **`silhouette_score`** menghitung $(b-a)/\\max(a,b)$ per titik, dengan $a$ = jarak rata-rata dalam cluster dan "
            "$b$ = jarak rata-rata ke cluster terdekat lainnya.")

    st.divider()
    st.subheader("Letak kode algoritma utama")
    st.markdown("Tabel ini dibaca langsung dari kode yang sedang berjalan. Jika modul anggota tim belum selesai, ditunjuk padanannya "
                "di `src/demo_data.py` (ditandai **demo**).")
    lok = provider.lokasi_kode()
    st.dataframe(pd.DataFrame([{"Tahap": x["tahap"], "Fungsi": x["fungsi"], "File": x["berkas"], "Baris": x["baris"],
                                "Sumber": "demo" if x["demo"] else "asli"} for x in lok]),
                 hide_index=True, width="stretch")
    pilih = st.selectbox("Tampilkan kode fungsi", [x["fungsi"] for x in lok], index=3,
                         help="Menampilkan kode sumber fungsi yang menjalankan tahap algoritma terpilih.")
    for x in lok:
        if x["fungsi"] == pilih:
            st.caption(f"{x['berkas']} · mulai baris {x['baris']}" + (" · **demo**" if x["demo"] else ""))
            st.code(x["kode"] or "# kode tidak tersedia", language="python")
    st.markdown("**Kode GUI dan visualisasi:** `app.py`, `pages/1_…5_*.py`, `src/plots.py` (grafik), `src/provider.py` "
                "(penghubung modul), `src/interpret.py` (interpretasi dan pesan error), `src/contoh.py` (contoh perhitungan).")

# ===========================================================================
# Tab 3: contoh perhitungan (live)
# ===========================================================================
with tab_contoh:
    st.subheader("Contoh perhitungan sederhana: 5 film, 2 fitur, 2 cluster")
    st.markdown("Semua angka di bawah **dihitung langsung** dari data contoh. Ubah nilai pada tabel, $m$, jumlah iterasi, atau α, "
                "lalu bandingkan dengan perhitungan kalkulator.")
    ca, cb = st.columns([3, 2])
    with ca:
        st.markdown("**Data contoh** (fitur sudah di skala 0–1; dapat diedit)")
        data_edit = st.data_editor(contoh.tabel_data_contoh(), hide_index=True, num_rows="fixed", key="contoh_data",
                                   width="stretch")
        st.markdown("**Keanggotaan awal $U^{(0)}$** (tiap baris dinormalkan agar berjumlah 1)")
        u0_df = pd.DataFrame(contoh.U0_CONTOH, columns=["Cluster 1", "Cluster 2"])
        u0_df.insert(0, "Film", contoh.NAMA_FILM)
        u0_edit = st.data_editor(u0_df, hide_index=True, num_rows="fixed", key="contoh_u0", width="stretch")
    with cb:
        m_c = st.slider("Fuzzifier m", 1.1, 3.0, 2.0, 0.1, key="contoh_m", help="Pangkat bobot keanggotaan.")
        n_it = st.slider("Jumlah iterasi yang ditampilkan", 1, 3, 2, key="contoh_iter", help="Berapa iterasi FCM dihitung.")
        alpha_c = st.slider("α untuk KNN fuzzy", 0.0, 1.0, 0.5, 0.1, key="contoh_alpha", help="Bobot jarak fitur pada KNN fuzzy.")

    X_c = data_edit[contoh.NAMA_FITUR].to_numpy(dtype=float)
    U0_c = u0_edit[["Cluster 1", "Cluster 2"]].to_numpy(dtype=float)
    galat = []
    if np.isnan(X_c).any() or np.isnan(U0_c).any():
        galat.append("Ada sel yang kosong. Isi semua angka pada tabel data dan keanggotaan awal.")
    elif ((X_c < 0) | (X_c > 1)).any():
        galat.append("Nilai fitur harus berada di antara 0 dan 1 (sudah dinormalisasi). Perbaiki sel yang di luar rentang.")
    elif (U0_c < 0).any() or (U0_c.sum(axis=1) <= 0).any():
        galat.append("Keanggotaan awal tidak boleh negatif dan tiap baris harus berjumlah lebih dari 0.")
    for g in galat:
        st.error(g)
    if not galat:
        langkah = contoh.langkah_fcm(X_c, U0_c, m_c, n_it)
        for L in langkah:
            with st.expander(f"Iterasi {L['iter']}: J_m = {L['J']:.4f} · Δ = {L['delta']:.4f}", expanded=(L["iter"] == 1)):
                st.markdown("**Langkah 1. Pusat cluster** $v_j = \\sum u^m x / \\sum u^m$ (rincian untuk cluster 1, fitur rating)")
                st.latex(contoh.teks_pusat(X_c, L["U_in"], m_c, 0, 0))
                v_df = pd.DataFrame(L["V"], columns=contoh.NAMA_FITUR, index=["Pusat C1", "Pusat C2"]).round(4)
                st.dataframe(v_df, width="stretch")
                st.markdown("**Langkah 2. Jarak Euclidean** $d_{ij}=\\lVert x_i - v_j\\rVert$")
                st.dataframe(pd.DataFrame(L["d"], columns=["d ke C1", "d ke C2"], index=contoh.NAMA_FILM).round(4), width="stretch")
                st.markdown("**Langkah 3. Keanggotaan baru** $u_{ij}=1/\\sum_k (d_{ij}/d_{ik})^{2/(m-1)}$ (rincian film A, cluster 1)")
                st.latex(contoh.teks_keanggotaan(L["d"], m_c, 0, 0, "A"))
                st.dataframe(pd.DataFrame(L["U_out"], columns=["u ke C1", "u ke C2"], index=contoh.NAMA_FILM).round(4), width="stretch")
                st.markdown(f"**Langkah 4. Fungsi objektif dan konvergensi:** $J_m = \\sum u^m d^2 = {L['J']:.4f}$; "
                            f"$\\max|U_{{baru}}-U_{{lama}}| = {L['delta']:.4f}$. "
                            "Proses lanjut jika nilai ini masih lebih besar dari ε.")
        if len(langkah) > 1:
            st.caption("J_m per iterasi: " + " → ".join(f"{L['J']:.4f}" for L in langkah) + " (turun menandakan proses benar).")
        V_list = [L["V"] for L in langkah]
        st.plotly_chart(plot_contoh_fcm(X_c, contoh.NAMA_FILM, V_list, [f"Setelah iterasi {L['iter']}" for L in langkah],
                                        langkah[-1]["U_out"]), width="stretch")

        # pemeriksaan: hasil tangan = fcm.membership
        try:
            u_modul, demo_m = provider._panggil("fcm", "membership", X_c[0], langkah[-1]["V"], m=m_c)
            cocok = np.allclose(u_modul, langkah[-1]["U_out"][0], atol=1e-6)
            st.caption(("✔ " if cocok else "⚠️ ") + "Keanggotaan film A hasil hitung tangan " +
                       ("sama" if cocok else "berbeda") + f" dengan `fcm.membership` ({'demo' if demo_m else 'modul asli'}).")
        except Exception as e:  # noqa: BLE001
            tampilkan_error(e, "Pemeriksaan dengan fcm.membership gagal.")

        st.divider()
        st.subheader("Dari keanggotaan FCM ke jarak KNN fuzzy")
        U_fin = langkah[-1]["U_out"]
        q = st.selectbox("Film acuan contoh", contoh.NAMA_FILM, index=2, key="contoh_q",
                         help="Film yang dicari tetangganya. Film C berada di antara dua kelompok.")
        qi = contoh.NAMA_FILM.index(q)
        pf = contoh.peringkat_knn_contoh(X_c, U_fin, qi, alpha_c)
        pb = contoh.peringkat_knn_contoh(X_c, U_fin, qi, 1.0)
        t = pf.iloc[0]
        nama_t = contoh.NAMA_FILM[int(t["idx"])]
        st.markdown(f"Vektor keanggotaan: $u_{q}$ = ({U_fin[qi, 0]:.3f}; {U_fin[qi, 1]:.3f}) dan $u_{nama_t}$ = "
                    f"({U_fin[int(t['idx']), 0]:.3f}; {U_fin[int(t['idx']), 1]:.3f}). Untuk pasangan **{q}–{nama_t}**:")
        st.latex(rf"d = {alpha_c:.1f}\times\underbrace{{{t['d_fitur']:.4f}}}_{{\lVert x_q-x_i\rVert}} + "
                 rf"{1 - alpha_c:.1f}\times\underbrace{{{t['d_mem']:.4f}}}_{{\lVert u_q-u_i\rVert}} = {t['d']:.4f}"
                 rf"\quad\Rightarrow\quad \text{{skor}} = \frac{{1}}{{1+{t['d']:.4f}}} = {t['skor']:.4f}")
        r1, r2 = st.columns(2)
        fmt = {"d_fitur": "{:.4f}", "d_mem": "{:.4f}", "d": "{:.4f}", "skor": "{:.4f}"}

        def _tampil(df):
            out = pd.DataFrame({"Peringkat": np.arange(1, len(df) + 1), "Film": [contoh.NAMA_FILM[int(i)] for i in df["idx"]],
                                "Jarak fitur": df["d_fitur"].round(4), "Jarak keanggotaan": df["d_mem"].round(4),
                                "d": df["d"].round(4), "Skor": df["skor"].round(4)})
            return out
        r1.markdown(f"**KNN fuzzy (α = {alpha_c:.1f})**")
        r1.dataframe(_tampil(pf), hide_index=True, width="stretch")
        r2.markdown("**KNN biasa (α = 1)**")
        r2.dataframe(_tampil(pb), hide_index=True, width="stretch")
        urut_f, urut_b = [int(i) for i in pf["idx"]], [int(i) for i in pb["idx"]]
        if urut_f != urut_b:
            st.success("Urutan tetangga **berubah** karena derajat keanggotaan ikut menentukan jarak.")
        else:
            st.info("Urutan sama untuk α ini. Turunkan α atau pilih film acuan lain (mis. C) untuk melihat pengaruh keanggotaan.")

# ===========================================================================
# Tab 4: arsitektur dan alur
# ===========================================================================
with tab_arsitek:
    ka, kb = st.columns(2)
    with ka:
        st.subheader("Arsitektur sistem")
        st.graphviz_chart(arsitektur_dot(), width="content")
    with kb:
        st.subheader("Flowchart algoritma")
        st.graphviz_chart(alur_algoritma_dot(), width="content")
    st.caption("Diagram arsitektur menggantikan diagram ANFIS pada daftar visualisasi tugas karena metode yang dipakai FCM + KNN.")

# ===========================================================================
# Tab 5: petunjuk dan anggota
# ===========================================================================
with tab_petunjuk:
    st.subheader("Petunjuk penggunaan")
    st.markdown(
        "1. **Input Data**: pilih *Dataset bawaan* atau unggah CSV berkolom wajib, lalu klik tombol muat. Opsional: masukkan film baru secara manual.\n"
        "2. **Proses**: atur c, m, ε, dan iterasi maksimum (atau pakai default), lalu klik **Jalankan Clustering**. Lihat parameter, grafik J_m, "
        "fungsi keanggotaan, hasil antara tiap iterasi, FPC, silhouette, dan heatmap.\n"
        "3. **Hasil**: pilih film acuan, atur k dan α, lalu klik **Cari Rekomendasi**. Baca interpretasi, tabel, uraian jarak FCM → KNN, dan peta cluster.\n"
        "4. **Evaluasi**: klik **Jalankan Pengujian** untuk membandingkan KNN fuzzy dengan KNN biasa (Precision@K, RMSE/MAE, pengaruh α).\n"
        "5. Jika muncul pesan merah, baca isinya: setiap pesan menjelaskan apa yang salah dan cara memperbaikinya."
    )
    with st.expander("Penjelasan lengkap: cara membaca grafik"):
        st.markdown(
            "- **J_m per iterasi**: turun lalu mendatar berarti FCM sudah konvergen.\n"
            "- **FPC dan silhouette per c**: pilih c dengan kedua nilai relatif tinggi.\n"
            "- **Fungsi keanggotaan**: tinggi kurva = derajat keanggotaan; perpotongan = batas antar cluster.\n"
            "- **Heatmap keanggotaan**: baris = film, kolom = cluster, gelap = keanggotaan tinggi.\n"
            "- **Perpindahan pusat cluster**: garis dari iterasi awal ke akhir; berhenti bergeser = konvergen.\n"
            "- **Peta cluster**: tiap titik adalah film; titik pucat berarti keanggotaannya terbagi.\n"
            "- **Uraian jarak**: segmen abu-abu = kontribusi fitur, merah = kontribusi keanggotaan FCM.\n"
            "- **Radar**: perbandingan ciri pusat cluster (rating, vote, popularitas, durasi, tahun).\n"
            "- **Precision@K, RMSE/MAE, sapuan α**: ukuran kualitas rekomendasi; makin tinggi precision dan makin rendah galat makin baik."
        )
    st.divider()
    st.subheader("Pembagian tugas dan kontribusi anggota")
    st.dataframe(pd.DataFrame([
        (1, "Anggota 1 (isi nama/NIM)", "Data dan preprocessing", "Pembersihan data, fitur, PCA genre, validasi CSV", "src/preprocessing.py, BAB I dan III"),
        (2, "Anggota 2 (isi nama/NIM)", "Fuzzy C-Means", "Implementasi FCM, pemilihan c, contoh hitung FCM", "src/fcm.py, BAB II dan IV"),
        (3, "Anggota 3 (isi nama/NIM)", "KNN, integrasi, evaluasi", "KNN berbobot fuzzy, Precision@K, contoh hitung KNN", "src/knn_fuzzy.py, src/evaluation.py"),
        (4, "Maytta (isi NIM)", "GUI dan visualisasi", "Enam halaman, grafik Plotly, interpretasi, contoh perhitungan, pesan error", "app.py, pages/, src/plots.py"),
    ], columns=["No", "Nama/NIM", "Tanggung jawab", "Kontribusi konkret", "Bukti pekerjaan"]).astype(str),
        hide_index=True, width="stretch")
    st.caption("Final Project Teknologi Logika Fuzzy · Informatika · Dosen: Dr. Made Agung Raharja, S.Si., M.Cs. · Kelompok 4 orang.")
