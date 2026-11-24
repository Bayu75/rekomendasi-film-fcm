"""Halaman Evaluasi (F-20 s.d. F-22). Anggota 4: GUI & Visualisasi."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import provider
from src.interpret import interpretasi_evaluasi
from src.plots import plot_actual_vs_pred, plot_alpha_sweep, plot_cluster_quality, plot_precision
from src.ui import butuh_model, page_setup, tampilkan_error

page_setup("Evaluasi", "Ukur kualitas cluster dan seberapa baik rekomendasi fuzzy dibanding KNN biasa.", aktif="evaluasi")
ss = st.session_state
if not butuh_model():
    st.stop()

# ---------------------------------------------------------------------------
# Bagian 1: protokol dan tombol (F-20)
# ---------------------------------------------------------------------------
st.subheader("Protokol pengujian")
st.markdown(
    "Sistem memilih **200 film acuan acak** (seed 42), mencari K rekomendasi dengan KNN fuzzy dan KNN biasa (α = 1), "
    "lalu menghitung **Precision@K**. Rekomendasi dianggap **relevan** jika berbagi minimal satu genre dengan film acuan."
)
st.dataframe(pd.DataFrame([
    ("1. Kualitas cluster", "FPC dan silhouette dari hasil FCM", "FPC mendekati 1, silhouette > 0,25"),
    ("2. Ketepatan rekomendasi", "Precision@5/10/20, KNN fuzzy vs KNN biasa (α = 1)", "Fuzzy ≥ KNN biasa"),
    ("3. Prediksi rating", "RMSE dan MAE: rating film acuan diprediksi dari rating tetangga", "Makin kecil makin baik"),
    ("4. Pengaruh α", "Precision@10 untuk α = 0; 0,25; 0,5; 0,75; 1", "Kontribusi keanggotaan FCM terlihat"),
], columns=["Skenario pengujian", "Metrik / cara ukur", "Harapan"]), hide_index=True, width="stretch")
alpha = st.slider("Bobot jarak fitur untuk KNN fuzzy (α)", 0.0, 1.0, 0.5, 0.1,
                  help="Nilai α yang dipakai KNN fuzzy. Baseline selalu α = 1 (KNN biasa).")
if st.button("Jalankan Pengujian", type="primary",
             help="Menghitung FPC, silhouette, dan Precision@5/10/20 untuk KNN fuzzy dan KNN biasa."):
    try:
        with st.spinner("Menjalankan pengujian pada 200 film acuan…"):
            provider.jalankan_pengujian(alpha=float(alpha))
        st.success("Pengujian selesai.")
    except Exception as e:  # noqa: BLE001
        tampilkan_error(e, "Pengujian gagal dijalankan.")

ev = ss.get("eval_hasil")
if ev is None:
    st.info("Klik Jalankan Pengujian untuk melihat metrik dan grafik.")
else:
    # -----------------------------------------------------------------------
    # Bagian 2: metrik dan grafik (F-21)
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Ringkasan metrik")
    prec = ev["precision"]
    baris10 = prec[prec["k"] == 10].iloc[0] if (prec["k"] == 10).any() else prec.iloc[len(prec) // 2]
    k_tampil = int(baris10["k"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("FPC", f"{ev['fpc']:.3f}", help="Fuzzy Partition Coefficient; mendekati 1 = partisi tegas.")
    m2.metric("Silhouette", f"{ev['silhouette']:.3f}", help="Kerapian cluster (berdasarkan cluster dominan). Target > 0,25.")
    m3.metric(f"Precision@{k_tampil} (fuzzy)", f"{baris10['fuzzy']:.3f}",
              delta=f"{baris10['fuzzy'] - baris10['baseline']:+.3f} vs KNN biasa")
    m4.metric(f"Precision@{k_tampil} (KNN biasa)", f"{baris10['baseline']:.3f}")
    tabel = prec.rename(columns={"k": "K", "fuzzy": "Precision KNN fuzzy", "baseline": "Precision KNN biasa"}).copy()
    tabel["Selisih"] = tabel["Precision KNN fuzzy"] - tabel["Precision KNN biasa"]
    st.dataframe(tabel, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tabel.columns if c != "K"})

    st.subheader("Precision@K: KNN fuzzy vs KNN biasa")
    st.plotly_chart(plot_precision(prec), width="stretch")

    st.subheader("Rating aktual vs prediksi sistem")
    st.markdown("Rating setiap film uji **diprediksi dari rating 10 tetangganya** (rata-rata berbobot skor kemiripan). "
                "Titik yang dekat garis putus-putus berarti sistem mengenali film yang benar-benar mirip.")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("RMSE KNN fuzzy", f"{ev['rmse_fuzzy']:.3f}", delta=f"{ev['rmse_fuzzy'] - ev['rmse_base']:+.3f} vs KNN biasa",
              delta_color="inverse", help="Akar rata-rata kuadrat galat; lebih kecil lebih baik.")
    e2.metric("MAE KNN fuzzy", f"{ev['mae_fuzzy']:.3f}", delta=f"{ev['mae_fuzzy'] - ev['mae_base']:+.3f} vs KNN biasa",
              delta_color="inverse", help="Rata-rata galat absolut; lebih kecil lebih baik.")
    e3.metric("RMSE KNN biasa", f"{ev['rmse_base']:.3f}")
    e4.metric("MAE KNN biasa", f"{ev['mae_base']:.3f}")
    st.plotly_chart(plot_actual_vs_pred(ev["aktual"], ev["pred_fuzzy"], ev["pred_base"], ev["judul_uji"]), width="stretch")

    st.subheader("Pengaruh α terhadap Precision@10")
    st.plotly_chart(plot_alpha_sweep(ev["sweep"]), width="stretch")
    st.caption("Titik α = 1 adalah KNN biasa. Jika kurva fuzzy berada di atas garis putus-putus, derajat keanggotaan FCM menambah ketepatan.")

    st.subheader("Kualitas tiap cluster")
    st.plotly_chart(plot_cluster_quality(ev["ukuran"], ev["silhouette_per"], ss.nama_cluster), width="stretch")

    # -----------------------------------------------------------------------
    # Bagian 3: interpretasi, kelebihan, keterbatasan (F-22)
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Interpretasi hasil")
    st.markdown(f"- Prediksi rating: RMSE {ev['rmse_fuzzy']:.3f} (fuzzy) vs {ev['rmse_base']:.3f} (KNN biasa); "
                f"MAE {ev['mae_fuzzy']:.3f} vs {ev['mae_base']:.3f}. Selisih kecil wajar karena rating tetangga sudah mirip "
                "pada kedua metode.")
    for poin in interpretasi_evaluasi(float(baris10["fuzzy"]), float(baris10["baseline"]), k_tampil,
                                      ev["fpc"], ev["silhouette"]):
        st.markdown(f"- {poin}")

st.divider()
st.subheader("Kelebihan dan keterbatasan")
kk1, kk2 = st.columns(2)
with kk1:
    st.markdown("**Kelebihan**\n\n"
                "- Film berciri campuran tetap tercatat lewat derajat keanggotaan.\n"
                "- Kontribusi FCM terukur dengan pembanding KNN biasa (α = 1).\n"
                "- Hasil bisa diulang karena seed dikunci.")
with kk2:
    st.markdown("**Keterbatasan**\n\n"
                "- Relevansi hanya berdasarkan genre yang sama, sehingga cenderung menguntungkan semua metode.\n"
                "- Silhouette bisa rendah karena data film padat dan bertumpuk.\n"
                "- Isi cerita (overview) tidak dipakai; hanya fitur numerik dan genre.\n"
                "- Prediksi rating memakai rating rata-rata (`vote_average`), bukan rating per pengguna dari `ratings_small.csv`.")

with st.expander("Bagaimana ini dihitung?"):
    st.latex(r"\mathrm{FPC} = \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{c} u_{ij}^{2}")
    st.latex(r"s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}} \qquad "
             r"\mathrm{Precision@K} = \frac{\text{jumlah rekomendasi relevan}}{K}")
    st.latex(r"\mathrm{RMSE} = \sqrt{\tfrac{1}{N}\sum (r_i - \hat r_i)^2} \qquad \mathrm{MAE} = \tfrac{1}{N}\sum |r_i - \hat r_i| \qquad "
             r"\hat r_q = \frac{\sum_{i \in kNN(q)} s_i\, r_i}{\sum_{i \in kNN(q)} s_i}")
    st.markdown("Silhouette memakai cluster dominan tiap film (keanggotaan terbesar) dan dihitung pada sampel hingga 3.000 film. "
                "Precision@K dirata-rata pada 200 film acuan acak.")
