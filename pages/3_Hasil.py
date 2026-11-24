"""Halaman Hasil (F-14 s.d. F-19, F-26). Anggota 4: GUI & Visualisasi."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import komponen, provider
from src.interpret import cluster_profile, fitur_baca, interpretasi_rekomendasi
from src.plots import (plot_cluster_radar, plot_clusters, plot_objective, plot_distance_breakdown, plot_membership_compare,
                       plot_query_membership, plot_rating_comparison)
from src.ui import butuh_model, page_setup, tampilkan_error

page_setup("Hasil Rekomendasi", "Temukan film serupa dengan bantuan Fuzzy C-Means dan KNN", aktif="hasil")
ss = st.session_state
if not butuh_model():
    st.stop()

FILM_BARU = "➕ Film baru dari input manual"
fv0 = ss.film_view
judul = fv0["title"].astype(str)
kembar = judul.str.lower().str.strip().duplicated(keep=False)
label = judul.where(~kembar, judul + " (" + fv0["_year"].round().astype("Int64").astype(str) + ")").tolist()
opsi = ([FILM_BARU] if ss.get("new_film") else []) + sorted(range(len(label)), key=lambda i: label[i].lower())

# ---------------------------------------------------------------------------
# Bagian 1: kontrol (F-14)
# ---------------------------------------------------------------------------
with st.container(key="kartu_kontrol"):
    st.html(komponen.judul_kartu("Pilih Film Acuan", "film"))
    kolom_a, kolom_b, kolom_c = st.columns([2, 1, 1])
    pilihan = kolom_a.selectbox("Film acuan", opsi, index=0,
                                format_func=lambda o: o if o == FILM_BARU else label[o],
                                help="Ketik untuk mencari judul. Hanya judul di dataset yang bisa dipilih, "
                                     "ditambah film baru jika sudah dibuat di halaman Input Data.")
    k = kolom_b.slider("Jumlah rekomendasi (k)", 5, 20, 10, help="Banyaknya film yang direkomendasikan (5–20).")
    alpha = kolom_c.slider("Bobot jarak fitur (α)", 0.0, 1.0, 0.5, 0.1,
                           help="α = 1: hanya jarak fitur (KNN biasa). α = 0: hanya jarak derajat keanggotaan. Default 0,5.")
    if alpha >= 1.0:
        st.info("Nilai α = 1 sama dengan KNN biasa tanpa fuzzy.")
    elif alpha <= 0.0:
        st.info("Nilai α = 0 hanya memakai kemiripan derajat keanggotaan; fitur asli tidak dipertimbangkan.")

    if st.button("Cari Rekomendasi", type="primary", help="Mencari k film terdekat dengan jarak gabungan fitur dan keanggotaan."):
        try:
            q = None if pilihan == FILM_BARU else int(pilihan)
            with st.spinner("Mencari film paling mirip…"):
                provider.cari_rekomendasi(q, k, alpha)
            ss.hasil_semua = False
        except Exception as e:  # noqa: BLE001
            tampilkan_error(e, "Rekomendasi tidak bisa dibuat.")

reco = ss.get("last_reco")
if reco is None:
    st.info("Pilih film acuan lalu klik Cari Rekomendasi.")
    st.stop()

U = ss.fcm_result["U"]
nama = ss.nama_cluster
fv = ss.film_view
u_q = reco["u_q"]
dom = int(np.argmax(u_q))
c_klaster = int(U.shape[1])

# data film acuan untuk kartu Film Pilihan
if reco["q_idx"] is None:
    fb = ss.new_film["film"]
    f_judul, f_rating, f_genre = reco["judul"], f"{float(fb['vote_average']):.1f}", list(fb["genres"])
    f_tahun, f_durasi, f_poster, f_sinopsis = int(fb["year"]), f"{int(fb['runtime'])} menit", None, None
    f_rating_num = float(fb["vote_average"])
else:
    r = fv.loc[reco["q_idx"]]
    f_judul, f_genre = reco["judul"], list(r["_genres"])
    f_rating_num = float(r["vote_average"])
    f_rating = f"{f_rating_num:.1f}"
    f_tahun = int(r["_year"]) if np.isfinite(r["_year"]) else None
    f_durasi = f"{int(r['runtime'])} menit" if "runtime" in r and np.isfinite(r["runtime"]) else None
    f_poster, f_sinopsis = r.get("poster_path"), r.get("overview")

tabel = reco["tabel"]

# ---------------------------------------------------------------------------
# Bagian 2: Ringkasan Hasil, Film Pilihan, empat kartu metrik (F-18)
# ---------------------------------------------------------------------------
st.html(komponen.callout_ringkasan(
    interpretasi_rekomendasi(reco["judul"], u_q, nama, reco["genres"], reco["rating_rata"], reco["alpha"])))
st.html(komponen.kartu_film_pilihan(
    f_judul, f_tahun, f_rating, f_durasi, f_genre, f_sinopsis, f_poster,
    [("Jumlah Rekomendasi (K)", str(reco["k"])), ("Alpha (α)", f"{reco['alpha']:.1f}".replace(".", ",")),
     ("Jumlah Cluster (c)", str(c_klaster)), ("Fuzzifier (m)", f"{ss.fcm_param['m']:g}".replace(".", ","))]))

mt = provider.metrik_ringkas()


def _angka(v, desimal=3):
    return "-" if v is None or not np.isfinite(v) else f"{v:.{desimal}f}".replace(".", ",")


st.html(komponen.baris_metrik([
    komponen.kartu_metrik("clapper", komponen.MERAH, _angka(float(tabel["Skor Kemiripan"].max())),
                          "Skor Kemiripan Tertinggi", tabel["Skor Kemiripan"].tolist(), "skor"),
    komponen.kartu_metrik("network", komponen.BIRU, _angka(mt["fpc"]), "FPC Score", mt["seri_fpc"], "fpc"),
    komponen.kartu_metrik("bars", komponen.HIJAU, _angka(mt["silhouette"]), "Silhouette Score", mt["seri_sil"], "sil"),
    komponen.kartu_metrik("target", komponen.AMBER, _angka(mt["precision10"]), "Precision@10", mt["seri_prec"], "prec"),
]))
if mt["precision10"] is None:
    st.caption("Precision@10 belum bisa dihitung; buka halaman Evaluasi dan klik Jalankan Pengujian.")

# ---------------------------------------------------------------------------
# Bagian 3: Daftar Rekomendasi (F-15, F-26)
# ---------------------------------------------------------------------------
semua = bool(ss.get("hasil_semua", False))
with st.container(key="kartu_daftar"):
    kiri, kanan = st.columns([4, 1], vertical_alignment="center")
    kiri.html(komponen.judul_kartu("Daftar Rekomendasi", "list"))
    if len(tabel) > 5:
        if kanan.button("Tampilkan Ringkas ←" if semua else "Lihat Semua →", key="lihat_semua", width="stretch",
                        help="Tampilkan 5 film teratas atau seluruh hasil rekomendasi."):
            ss.hasil_semua = not semua
            st.rerun()
    tampil = tabel if semua else tabel.head(5)
    poster_rec = fv.iloc[reco["idx"]]["poster_path"].tolist() if "poster_path" in fv.columns else [None] * len(tabel)
    st.html(komponen.tabel_rekomendasi([
        {"no": int(b["No"]), "judul": b["Judul"], "tahun": b["Tahun"] if pd.notna(b["Tahun"]) else "-",
         "genre": b["Genre"], "skor": b["Skor Kemiripan"], "poster_path": poster_rec[i]}
        for i, (_, b) in enumerate(tampil.iterrows())]))
    st.caption(f"Menampilkan {len(tampil)} dari {len(tabel)} film. Skor kemiripan = 1 / (1 + jarak); makin dekat ke 1 makin mirip.")

# ---------------------------------------------------------------------------
# Bagian 4: Visualisasi Cluster dan Fungsi Objektif FCM (F-16, F-17)
# ---------------------------------------------------------------------------
X2d, V2d = provider.koordinat_2d()
fb_xy = ss.new_film["xy"] if reco["q_idx"] is None else None
bawah1, bawah2 = st.columns(2)
with bawah1, st.container(key="kartu_cluster"):
    st.html(komponen.judul_kartu("Visualisasi Cluster", "scatter"))
    fig_c = plot_clusters(X2d, U, V2d, q_idx=reco["q_idx"], rec_idx=reco["idx"], names=[f"C{j + 1}" for j in range(c_klaster)],
                          hover=provider.hover_df(), query_xy=fb_xy, query_label="Film acuan")
    fig_c.update_layout(title=None, height=330, margin=dict(l=40, r=10, t=10, b=70),
                        legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=11)))
    st.plotly_chart(fig_c, width="stretch")
with bawah2, st.container(key="kartu_objektif"):
    st.html(komponen.judul_kartu("Fungsi Objektif FCM", "trend"))
    fig_j = plot_objective(ss.fcm_result.get("J_history", []))
    fig_j.update_layout(title=None, height=330, margin=dict(l=55, r=10, t=10, b=55), showlegend=False)
    st.plotly_chart(fig_j, width="stretch")
with st.expander("Arti warna C1–C5 (nama cluster)"):
    st.markdown("\n".join(f"- **C{j + 1}**: {n}" for j, n in enumerate(nama)))
st.caption("Warna = cluster dominan; titik makin pucat berarti keanggotaannya makin terbagi. Bintang putih = film acuan, "
           "lingkaran merah = rekomendasi, tanda × = pusat cluster.")

# ---------------------------------------------------------------------------
# Tabel lengkap dan unduh CSV
# ---------------------------------------------------------------------------
with st.container(key="kartu_tabel"):
    st.html(komponen.judul_kartu("Tabel Lengkap dan Unduh", "database"))
    st.dataframe(
        tabel, hide_index=True, width="stretch",
        column_config={
            "Tahun": st.column_config.NumberColumn(format="%d"),
            "Rating": st.column_config.NumberColumn(format="%.1f"),
            "Skor Kemiripan": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f",
                                                              help="1 / (1 + jarak). Makin dekat ke 1 makin mirip."),
            "Keanggotaan": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f",
                                                           help="Derajat keanggotaan film di cluster dominannya."),
        })
    st.download_button("Unduh Hasil (CSV)", tabel.to_csv(index=False).encode("utf-8"),
                       file_name="rekomendasi_fuzzyflix.csv", mime="text/csv",
                       help="Menyimpan tabel rekomendasi di atas sebagai file CSV.")

# ---------------------------------------------------------------------------
# Bagian 3b: integrasi FCM → KNN (brief D): derajat keanggotaan masuk ke rumus jarak
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Integrasi FCM → KNN: dari mana skor ini berasal")
rin = reco["rincian"]
a_ = reco["alpha"]
st.markdown(
    "KNN di FuzzyFlix **tidak memakai label cluster**. Vektor derajat keanggotaan hasil FCM, $u_q$ dan $u_i$, "
    "dimasukkan langsung ke dalam jarak. Film dengan pola keanggotaan mirip jadi lebih dekat walaupun fiturnya sedikit berbeda."
)
st.latex(r"d(q,i) = \underbrace{\alpha\,\lVert x_q - x_i\rVert}_{\text{jarak fitur}} + "
         r"\underbrace{(1-\alpha)\,\lVert u_q - u_i\rVert}_{\text{jarak keanggotaan FCM}}")
r0 = rin.iloc[0]
st.markdown(
    f"**Contoh untuk rekomendasi no. 1 ({reco['judul_rec'][0]}):** "
    f"$d = {a_:.1f} \\times {r0['d_fitur']:.3f} + {1 - a_:.1f} \\times {r0['d_mem']:.3f} = {r0['d']:.3f}$, "
    f"sehingga skor kemiripan $= 1/(1+{r0['d']:.3f}) = {r0['skor']:.3f}$."
)
if reco["selisih_jarak"] > 1e-6:
    st.warning(f"Jarak yang dihitung ulang di GUI berbeda {reco['selisih_jarak']:.4f} dari keluaran modul KNN. "
               "Periksa apakah `knn_fuzzy.recommend` memakai rumus PRD 5.3 (norma Euclidean).")
else:
    st.caption("✔ Jarak yang dihitung ulang di GUI dari X dan U sama dengan jarak keluaran modul KNN.")

ti1, ti2 = st.columns(2)
with ti1:
    st.plotly_chart(plot_distance_breakdown(rin, reco["judul_rec"], a_), width="stretch")
with ti2:
    U_pakai = np.vstack([u_q, U[reco["idx"]]])
    label = [f"★ {reco['judul']}"] + [f"{i + 1}. {t}" for i, t in enumerate(reco["judul_rec"])]
    st.plotly_chart(plot_membership_compare(U_pakai, label, nama), width="stretch")

tab_rin = pd.DataFrame({"No": np.arange(1, len(rin) + 1), "Judul": reco["judul_rec"],
                        "‖x_q − x_i‖": rin["d_fitur"], "‖u_q − u_i‖": rin["d_mem"],
                        f"α × fitur": rin["bagian_fitur"], f"(1−α) × keanggotaan": rin["bagian_mem"],
                        "Jarak d": rin["d"], "Skor 1/(1+d)": rin["skor"]})
with st.expander("Tabel uraian jarak"):
    st.dataframe(tab_rin, hide_index=True, width="stretch",
                 column_config={c_: st.column_config.NumberColumn(format="%.3f") for c_ in tab_rin.columns[2:]})

st.markdown("**Apa yang berubah dibanding KNN biasa (α = 1)?**")
judul_semua = fv["title"].astype(str).to_numpy()
set_f, set_b = set(map(int, reco["idx"])), set(map(int, reco["idx_base"]))
masuk, keluar = sorted(set_f - set_b), sorted(set_b - set_f)
ov1, ov2, ov3 = st.columns(3)
ov1.metric("Rekomendasi yang sama", f"{len(set_f & set_b)} dari {len(set_f)}")
ov2.metric("Masuk karena keanggotaan FCM", len(masuk))
ov3.metric("Keluar dibanding KNN biasa", len(keluar))
if masuk or keluar:
    st.markdown("- **Masuk karena derajat keanggotaan:** " + (", ".join(judul_semua[masuk]) if masuk else "-") +
                "\n- **Tersingkir (ada di KNN biasa saja):** " + (", ".join(judul_semua[keluar]) if keluar else "-"))
else:
    st.caption("Untuk film acuan dan α ini, hasil KNN fuzzy sama dengan KNN biasa. Coba turunkan α (mis. 0,2) agar pengaruh keanggotaan terlihat.")

st.plotly_chart(plot_rating_comparison(
    float(fb["vote_average"]) if reco["q_idx"] is None else float(fv.loc[reco["q_idx"], "vote_average"]),
    reco["judul_rec"], reco["ratings"], reco["judul"]), width="stretch")
st.caption("Perbandingan rating film acuan dengan film yang direkomendasikan: rating yang berdekatan menandakan rekomendasi yang selaras.")

# ---------------------------------------------------------------------------
# Bagian 5: keanggotaan dan ciri cluster (F-17, F-19)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Keanggotaan dan ciri cluster")
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(plot_query_membership(u_q, nama, title=f"Keanggotaan: {reco['judul']}"), width="stretch")
with g2:
    st.plotly_chart(plot_cluster_radar(cluster_profile(U, fv, ss.fcm_param["m"]), nama, ref=fitur_baca(fv)),
                    width="stretch")

with st.expander("Bagaimana ini dihitung?"):
    st.markdown("KNN mengukur jarak gabungan antara fitur film dan derajat keanggotaannya. "
                "Film acuan tidak ikut muncul di hasil.")
    st.latex(r"d(q,i) = \alpha\,\lVert x_q - x_i\rVert + (1-\alpha)\,\lVert u_q - u_i\rVert")
    st.latex(r"\text{skor kemiripan} = \frac{1}{1 + d}")
    st.markdown("Untuk film baru, $u_q$ dihitung dengan rumus keanggotaan FCM memakai pusat cluster $V$ yang tersimpan. "
                "Radar menampilkan ciri pusat cluster dalam satuan asli yang diskalakan 0–1 terhadap seluruh film.")
