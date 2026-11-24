"""Halaman Proses (F-08 s.d. F-13, F-25). Anggota 4: GUI & Visualisasi."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import provider
from src.interpret import fmt_int, validasi_parameter
from src.plots import (plot_center_trajectory, plot_membership_curve, plot_membership_function,
                       plot_membership_heatmap, plot_objective, plot_pca_variance, plot_select_c)
from src.ui import butuh_data, page_setup, tampilkan_error

page_setup("Proses", "Atur parameter Fuzzy C-Means, jalankan clustering, dan amati konvergensinya.", aktif="proses")
ss = st.session_state
if not butuh_data():
    st.stop()

# ---------------------------------------------------------------------------
# Bagian 1: penjelasan tahapan (F-10)
# ---------------------------------------------------------------------------
st.subheader("Tahapan proses")
t1, t2, t3, t4 = st.columns(4)
t1.markdown("**1. Inisialisasi**\n\nPilih pusat cluster awal dari film acak, lalu hitung derajat keanggotaan awal.")
t2.markdown("**2. Hitung pusat cluster**\n\nPusat tiap cluster = rata-rata fitur film, dibobot derajat keanggotaan.")
t3.markdown("**3. Perbarui keanggotaan**\n\nFilm yang dekat dengan sebuah pusat mendapat derajat keanggotaan lebih besar di cluster itu.")
t4.markdown("**4. Cek konvergensi**\n\nUlangi langkah 2–3 sampai perubahan keanggotaan lebih kecil dari ε atau iterasi mencapai batas.")

st.info("**Keluaran tahap ini (matriks keanggotaan U dan pusat cluster V) langsung dipakai KNN** di halaman Hasil: "
        "vektor keanggotaan setiap film menjadi bagian dari rumus jarak, bukan sekadar label cluster.")

st.divider()

# ---------------------------------------------------------------------------
# Bagian 2: parameter dan tombol (F-08, F-09)
# ---------------------------------------------------------------------------
st.subheader("Parameter clustering")
param = ss.get("fcm_param") or provider.DEFAULT_PARAM
saran = provider.saran_c(ss.select_c_df) if ss.get("select_c_df") is not None else None
for kunci, nilai in (("p_c", param["c"]), ("p_m", param["m"]), ("p_eps", param["error"]), ("p_iter", param["maxiter"])):
    ss.setdefault(kunci, nilai)


def _reset_param() -> None:
    d = provider.DEFAULT_PARAM
    ss.p_c, ss.p_m, ss.p_eps, ss.p_iter = d["c"], d["m"], d["error"], d["maxiter"]


p1, p2, p3, p4 = st.columns(4)
c = p1.number_input("Jumlah cluster (c)", 2, 10, key="p_c", step=1,
                    help="Banyaknya kelompok film (2–10). Gunakan grafik pemilihan c di bawah sebagai acuan.")
m = p2.slider("Fuzzifier (m)", 1.1, 3.0, key="p_m", step=0.1,
              help="Tingkat 'kabur' keanggotaan. m mendekati 1 = hampir tegas, m besar = keanggotaan makin merata. Default 2,0.")
eps = p3.select_slider("Toleransi (ε)", options=[1e-6, 1e-5, 1e-4, 1e-3], key="p_eps",
                       format_func=lambda v: f"{v:.0e}",
                       help="Proses berhenti jika perubahan keanggotaan terbesar < ε. Makin kecil makin teliti tetapi lebih lama.")
maxiter = p4.number_input("Iterasi maksimum", 50, 1000, key="p_iter", step=50,
                          help="Batas jumlah iterasi bila konvergensi belum tercapai (50–1000).")
if saran is not None:
    st.caption(f"Saran c dari tabel pemilihan di bawah (peringkat gabungan FPC dan silhouette): **c = {saran}**.")

tb1, tb2, tb3 = st.columns([1, 1, 1])
jalan = tb1.button("Jalankan Clustering", type="primary",
                   help="Menjalankan Fuzzy C-Means pada seluruh film dengan parameter di atas (seed tetap 42).")
tb2.button("Reset Parameter", on_click=_reset_param, help="Kembalikan c, m, ε, dan iterasi ke nilai awal.")
hitung_c = tb3.button("Hitung Pemilihan c", help="Menghitung FPC dan silhouette untuk c = 3 sampai 8 (bisa memakan waktu).")

if jalan:
    galat = validasi_parameter(int(c), float(m), float(eps), int(maxiter))
    if galat:
        for g in galat:
            st.error(g)
    else:
        try:
            with st.spinner("Menjalankan Fuzzy C-Means…"):
                hasil = provider.jalankan_clustering(int(c), float(m), float(eps), int(maxiter))
            st.success(f"Clustering selesai dalam {hasil['n_iter']} iterasi (FPC = {hasil['fpc']:.3f}).")
        except Exception as e:  # noqa: BLE001
            tampilkan_error(e, "Clustering gagal dijalankan.")
if hitung_c:
    try:
        with st.spinner("Menghitung FPC dan silhouette untuk c = 3…8…"):
            provider.hitung_pilih_c(float(m), float(eps), int(maxiter))
        saran = provider.saran_c(ss.select_c_df)
        st.success(f"Pemilihan c selesai. Saran: c = {saran}.")
    except Exception as e:  # noqa: BLE001
        tampilkan_error(e, "Pemilihan c gagal dihitung.")

if not provider.model_siap():
    st.warning("Model belum dijalankan. Klik Jalankan Clustering untuk melihat hasil dan grafik.")
    st.stop()

# ---------------------------------------------------------------------------
# Bagian 3: ringkasan dan grafik (F-11, F-12, F-13, F-25)
# ---------------------------------------------------------------------------
hasil = ss.fcm_result
prm = ss.fcm_param
st.divider()
st.subheader("Ringkasan hasil")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Jumlah cluster (c)", prm["c"])
r2.metric("FPC", f"{hasil['fpc']:.3f}", help="Fuzzy Partition Coefficient: mendekati 1 berarti partisi tegas, mendekati 1/c berarti sangat kabur.")
r3.metric("Iterasi", hasil["n_iter"] if hasil.get("n_iter") is not None else "-")
J = hasil.get("J_history") or []
r4.metric("J_m akhir", f"{J[-1]:.2f}" if len(J) else "-")
if not len(J):
    st.caption("Model dimuat dari `models/`, sehingga riwayat J_m belum ada. Klik Jalankan Clustering untuk melihatnya.")

st.subheader("Parameter yang digunakan pada proses ini")
st.dataframe(pd.DataFrame([
    ("c (jumlah cluster)", prm["c"], "Banyaknya pusat cluster V yang dicari."),
    ("m (fuzzifier)", f"{prm['m']:.1f}", "Pangkat bobot keanggotaan; m → 1 hampir tegas, m besar makin merata."),
    ("ε (toleransi)", f"{prm['error']:.0e}", "Berhenti jika max|U_baru − U_lama| < ε."),
    ("Iterasi maksimum", prm["maxiter"], "Batas iterasi bila belum konvergen."),
    ("Seed", 42, "Menetapkan inisialisasi awal sehingga hasil dapat diulang."),
    ("Jarak", "Euclidean", "d_ij = ‖x_i − v_j‖ pada fitur ternormalisasi (numerik + PCA genre)."),
    ("Data", f"{fmt_int(ss.X.shape[0])} film × {ss.X.shape[1]} fitur", "Ukuran matriks X yang dikelompokkan."),
    ("Iterasi sampai konvergen", hasil["n_iter"] if hasil.get("n_iter") is not None else "-", "Dilaporkan oleh modul FCM."),
], columns=["Parameter", "Nilai", "Keterangan"]).astype(str), hide_index=True, width="stretch")

st.subheader("Konvergensi dan pemilihan jumlah cluster")
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(plot_objective(J), width="stretch")
    st.caption("J_m seharusnya turun lalu mendatar; itu tanda proses sudah konvergen.")
with g2:
    if ss.get("select_c_df") is not None:
        st.plotly_chart(plot_select_c(ss.select_c_df, c_selected=prm["c"]), width="stretch")
        st.caption("Pilih c dengan FPC tinggi dan silhouette tinggi. Garis putus merah = c yang sedang dipakai.")
    else:
        st.info("Klik Hitung Pemilihan c untuk melihat FPC dan silhouette tiap c.")

st.divider()
st.subheader("Fungsi keanggotaan FCM")
st.markdown("Grafik ini adalah **fungsi keanggotaan** yang dipelajari FCM: seberapa besar derajat keanggotaan terhadap tiap cluster "
            "saat satu fitur digeser dari terkecil ke terbesar, sedangkan fitur lain ditahan pada rata-rata seluruh film. "
            "Tab kedua memperlihatkan rata-rata keanggotaan yang benar-benar teramati pada data.")
nama = provider.nama_fitur()
pilih = st.selectbox("Fitur", nama, index=0, key="fitur_kurva",
                     help="Pilih fitur yang digeser. Perpotongan kurva menunjukkan batas antar cluster, sedangkan "
                          "daerah tumpang tindih adalah film berciri campuran.")
tab_a, tab_b = st.tabs(["Fungsi keanggotaan (hitungan eksak)", "Rata-rata keanggotaan pada data"])
with tab_a:
    try:
        grid, u_grid, ref_val = provider.kurva_keanggotaan(nama.index(pilih))
        st.plotly_chart(plot_membership_function(grid, u_grid, pilih, ss.nama_cluster, x_marker=ref_val), width="stretch")
    except Exception as e:  # noqa: BLE001
        tampilkan_error(e, "Kurva keanggotaan tidak bisa dihitung.")
with tab_b:
    st.plotly_chart(plot_membership_curve(ss.X, hasil["U"], nama.index(pilih), feature_label=pilih,
                                          cluster_names=ss.nama_cluster), width="stretch")

st.divider()
st.subheader("Hasil antara: keanggotaan dan pusat cluster per iterasi")
snap = ss.get("hasil_antara")
if not snap:
    st.info("Hasil antara belum tersedia. Klik Jalankan Clustering agar U dan V tiap iterasi bisa ditampilkan.")
else:
    st.markdown("U dan V di bawah diambil setelah iterasi tertentu (seed sama). Lihat bagaimana keanggotaan 8 film pertama "
                "menajam dan pusat cluster bergeser sampai konvergen.")
    pilih_s = st.multiselect("Snapshot yang ditampilkan", [x["label"] for x in snap],
                             default=[snap[0]["label"], snap[-1]["label"]] if len(snap) > 1 else [snap[0]["label"]],
                             help="Pilih satu atau beberapa titik iterasi untuk dibandingkan.")
    n_tampil = 8
    judul_film = ss.film_view["title"].astype(str).tolist()[:n_tampil]
    kolom_c = [f"C{j + 1}" for j in range(prm["c"])]
    for x in snap:
        if x["label"] not in pilih_s:
            continue
        st.markdown(f"**{x['label']}** · J_m = {x['J']:.3f}" if pd.notna(x["J"]) else f"**{x['label']}**")
        k1 = k2 = st
        tu = pd.DataFrame(x["U"][:n_tampil], columns=kolom_c)
        tu.insert(0, "Film", judul_film)
        k1.dataframe(tu, hide_index=True, width="stretch",
                     column_config={c_: st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f") for c_ in kolom_c})
        tv = pd.DataFrame(x["V"][:, :5], columns=[provider.nama_fitur()[i] for i in range(min(5, ss.X.shape[1]))])
        tv.insert(0, "Pusat", kolom_c)
        k2.dataframe(tv.round(3), hide_index=True, width="stretch")
        k2.caption("Pusat cluster V (5 fitur pertama, ternormalisasi).")
    X2d, _ = provider.koordinat_2d()
    st.plotly_chart(plot_center_trajectory(X2d, [provider._proyeksi(x["V"]) for x in snap],
                                           [x["label"] for x in snap], hasil["U"], ss.nama_cluster), width="stretch")
    st.caption("Garis menghubungkan posisi pusat tiap cluster dari iterasi awal sampai akhir (tanda ×). "
               "Pusat yang tidak lagi bergeser menandakan konvergensi.")

st.divider()
st.subheader("Matriks keanggotaan (sampel 20 film)")
st.plotly_chart(plot_membership_heatmap(hasil["U"], ss.film_view["title"].astype(str).tolist(), n=20,
                                        cluster_names=ss.nama_cluster), width="stretch")
st.caption("Tiap baris adalah satu film; jumlah nilai di satu baris = 1. Baris dengan dua kotak gelap adalah film berciri campuran.")

st.divider()
st.subheader("Detail fitur dan variansi")
d1, d2 = st.columns(2)
with d1:
    pca = (ss.artefak or {}).get("pca")
    if pca is not None and hasattr(pca, "explained_variance_ratio_"):
        st.plotly_chart(plot_pca_variance(pca), width="stretch")
    else:
        st.info("Objek PCA genre tidak ditemukan di artefak (`artefak['pca']`), sehingga grafik variansi tidak ditampilkan.")
with d2:
    st.caption("PCA dipakai untuk meringkas 20 kolom genre menjadi beberapa komponen sebelum clustering.")

with st.expander("Bagaimana ini dihitung?"):
    st.markdown("Fuzzy C-Means meminimalkan fungsi objektif berikut dengan memperbarui pusat cluster dan derajat keanggotaan bergantian.")
    st.latex(r"J_m = \sum_{i=1}^{n}\sum_{j=1}^{c} u_{ij}^{m}\, d_{ij}^{2}")
    st.latex(r"v_j = \frac{\sum_i u_{ij}^{m}\, x_i}{\sum_i u_{ij}^{m}} \qquad "
             r"u_{ij} = \frac{1}{\sum_{k=1}^{c}\left(d_{ij}/d_{ik}\right)^{2/(m-1)}}")
    st.markdown("Proses berhenti jika $\\max|U_{baru}-U_{lama}| < \\varepsilon$ atau iterasi mencapai batas. "
                "FPC dihitung sebagai $\\frac{1}{n}\\sum_i\\sum_j u_{ij}^2$.")
