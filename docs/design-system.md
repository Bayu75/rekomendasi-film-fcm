# Design System – FuzzyFlix

> Panduan tampilan untuk GUI Streamlit FuzzyFlix. Tujuannya supaya keenam halaman terlihat konsisten walaupun dikerjakan orang berbeda.
> Status: Draft v1 (21 Nov 2026). **[Diputuskan]** = sudah disepakati, **[Usulan]** = boleh diubah tim.

---

## 1. Prinsip

1. **Jelas dulu, baru cantik.** Dosen harus bisa mengikuti alur input → proses → hasil → evaluasi tanpa penjelasan lisan.
2. **Proses fuzzy selalu terlihat.** Setiap hasil menampilkan derajat keanggotaan, bukan hanya label cluster.
3. **Konsisten.** Warna cluster, istilah, dan posisi tombol sama di semua halaman.
4. **Ramah kesalahan.** Input salah selalu dijawab pesan yang menjelaskan apa yang salah dan cara memperbaikinya.

---

## 2. Warna

### 2.1 Warna dasar **[Usulan]** (tema gelap ala aplikasi film)

| Token | Hex | Pemakaian |
|---|---|---|
| `primary` | `#E50914` | Tombol utama, aksen, link aktif |
| `primary-hover` | `#B20710` | Hover tombol utama |
| `bg` | `#0F1115` | Latar halaman |
| `surface` | `#1A1D24` | Kartu, sidebar, kontainer |
| `border` | `#2A2F3A` | Garis kartu, pemisah tabel |
| `text` | `#F5F5F7` | Teks utama |
| `text-muted` | `#A0A6B1` | Keterangan, caption |
| `success` | `#22C55E` | Pesan berhasil |
| `warning` | `#F59E0B` | Peringatan |
| `error` | `#EF4444` | Pesan error |
| `info` | `#3B82F6` | Info dan tips |

Pasang di `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#E50914"
backgroundColor = "#0F1115"
secondaryBackgroundColor = "#1A1D24"
textColor = "#F5F5F7"
font = "sans serif"
```

### 2.2 Warna cluster **[Diputuskan]**

Dipakai di **semua** grafik (scatter, bar keanggotaan, radar, heatmap legend). Cluster ke-n selalu warna yang sama.

| Cluster | Hex |
|---|---|
| C1 | `#4C78A8` |
| C2 | `#F58518` |
| C3 | `#54A24B` |
| C4 | `#E45756` |
| C5 | `#72B7B2` |
| C6 | `#EECA3B` |
| C7 | `#B279A2` |
| C8 | `#9D755D` |

```python
# src/plots.py
CLUSTER_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756",
                  "#72B7B2", "#EECA3B", "#B279A2", "#9D755D"]
```

Warna khusus:
- Film acuan: bintang `#FFFFFF` dengan outline `#E50914`, ukuran 18.
- Film rekomendasi: lingkaran outline `#E50914`, tanpa isi, ukuran 12.
- Pusat cluster: simbol `x` warna cluster, ukuran 16, garis tebal.
- Baseline (KNN biasa): `#A0A6B1`, garis/bar putus-putus atau abu-abu.
- Heatmap keanggotaan: skala `Blues` (0 = terang, 1 = gelap).

---

## 3. Tipografi

Pakai font bawaan Streamlit (sans serif). Hirarki:

| Elemen | Fungsi Streamlit | Contoh |
|---|---|---|
| Judul halaman | `st.title()` | "Hasil Rekomendasi" |
| Judul bagian | `st.subheader()` | "Derajat Keanggotaan Film Acuan" |
| Teks isi | `st.markdown()` | Penjelasan 1–3 kalimat |
| Keterangan kecil | `st.caption()` | Satuan, sumber, catatan |
| Rumus | `st.latex()` | Rumus FCM dan KNN |
| Angka penting | `st.metric()` | FPC, silhouette, Precision@10 |

Aturan:
- Satu `st.title` per halaman.
- Penjelasan maksimal 3 kalimat per blok; detail panjang masuk `st.expander("Penjelasan lengkap")`.
- Angka desimal: 3 digit untuk keanggotaan dan metrik (`0.662`), 1 digit untuk rating (`7.8`).

---

## 4. Layout

### 4.1 Kerangka halaman **[Diputuskan]**

```
┌───────────────┬──────────────────────────────────────┐
│ Sidebar       │ st.title                             │
│ - Logo/nama   │ caption: 1 kalimat tujuan halaman    │
│ - Navigasi    │──────────────────────────────────────│
│ - Status data │ Bagian 1 (input/kontrol)             │
│   & model     │ Bagian 2 (hasil/grafik)              │
│               │ Bagian 3 (interpretasi/penjelasan)   │
└───────────────┴──────────────────────────────────────┘
```

- `st.set_page_config(page_title="FuzzyFlix", page_icon="🎬", layout="wide")` di setiap halaman.
- Sidebar menampilkan status: ✅ Data dimuat (9.089 film) · ✅ Model FCM (c = 5) atau ⚠️ Model belum dijalankan.
- Kontrol (parameter, pilihan film) ada di atas, hasil di bawah. Bukan di sidebar, supaya terlihat saat demo.

### 4.2 Grid
- Dua kolom grafik: `col1, col2 = st.columns(2)`.
- Metrik: `st.columns(3)` atau `st.columns(4)` berisi `st.metric`.
- Form parameter: `st.columns(4)` untuk c, m, ε, maxiter.

### 4.3 Jarak
- Pisahkan bagian dengan `st.divider()`.
- Jangan menumpuk lebih dari 2 grafik besar tanpa judul bagian di antaranya.

---

## 5. Komponen

### 5.1 Tombol
| Jenis | Kode | Pemakaian |
|---|---|---|
| Utama | `st.button("Jalankan Clustering", type="primary")` | Satu per halaman, aksi utama |
| Sekunder | `st.button("Reset Parameter")` | Aksi tambahan |

Label tombol = kata kerja + objek: "Jalankan Clustering", "Cari Rekomendasi", "Jalankan Pengujian", "Unggah Dataset".

### 5.2 Input
| Data | Komponen | Batas |
|---|---|---|
| Film acuan | `st.selectbox` dengan pencarian | Hanya judul di dataset |
| k | `st.slider` | 5–20, default 10 |
| α | `st.slider` | 0.0–1.0, step 0.1, default 0.5 |
| c | `st.number_input` | 2–10 |
| m | `st.slider` | 1.1–3.0, default 2.0 |
| ε | `st.select_slider` | 1e-6, 1e-5, 1e-4, 1e-3 |
| maxiter | `st.number_input` | 50–1000, default 300 |
| Genre (manual) | `st.multiselect` | Min 1 genre |
| Angka manual | `st.number_input` dengan `min_value`/`max_value` | Sesuai rentang dataset |
| Dataset | `st.file_uploader(type="csv")` | Maks 200 MB |

Setiap input diberi `help="..."` berisi penjelasan singkat variabel.

### 5.3 Pesan
| Situasi | Komponen | Contoh teks |
|---|---|---|
| Berhasil | `st.success` | "Clustering selesai dalam 42 iterasi (FPC = 0.612)." |
| Input salah | `st.error` | "Kolom `runtime` tidak ditemukan. Pastikan CSV memiliki kolom: id, title, …" |
| Peringatan | `st.warning` | "Model belum dijalankan. Buka halaman Proses lalu klik Jalankan Clustering." |
| Info | `st.info` | "Nilai α = 1 sama dengan KNN biasa tanpa fuzzy." |
| Proses berjalan | `st.spinner` | "Menjalankan Fuzzy C-Means…" |

Format pesan error: **apa yang salah** + **cara memperbaiki**. Hindari pesan teknis seperti traceback.

### 5.4 Tabel
- Pakai `st.dataframe(df, hide_index=True, use_container_width=True)`.
- Kolom tabel rekomendasi (urutan tetap): `No · Judul · Tahun · Genre · Rating · Skor Kemiripan · Cluster Dominan · Keanggotaan`.
- Skor dan keanggotaan tampil sebagai progress bar lewat `st.column_config.ProgressColumn` (0–1).

### 5.5 Kartu metrik
```python
c1, c2, c3 = st.columns(3)
c1.metric("FPC", "0.612")
c2.metric("Silhouette", "0.287")
c3.metric("Precision@10", "0.84", delta="+0.06 vs KNN biasa")
```
*(angka di atas contoh, bukan hasil)*

### 5.6 Expander penjelasan
Setiap halaman Proses, Hasil, dan Evaluasi punya `st.expander("Bagaimana ini dihitung?")` berisi rumus (`st.latex`) dan 2–4 kalimat penjelasan.

---

## 6. Visualisasi

### 6.1 Aturan umum **[Diputuskan]**
- Library: **Plotly** untuk grafik interaktif di GUI (`st.plotly_chart(fig, use_container_width=True)`); Matplotlib boleh untuk gambar laporan.
- Semua fungsi grafik ada di `src/plots.py` dan me-return figure; halaman hanya menampilkan.
- Template Plotly: `template="plotly_dark"`, latar transparan.
- Setiap grafik punya judul sumbu dengan satuan dan legend jika lebih dari satu seri.
- Tinggi standar: 420 px (grafik utama 520 px).

### 6.2 Daftar grafik

| Grafik | Fungsi | Tipe | Halaman |
|---|---|---|---|
| Fungsi objektif per iterasi | `plot_objective(J_history)` | Line | Proses |
| FPC & silhouette per c | `plot_select_c(df)` | Line 2 seri, titik c terpilih ditandai | Proses |
| Heatmap keanggotaan | `plot_membership_heatmap(U, titles, n=20)` | Heatmap `Blues` | Proses |
| Explained variance PCA | `plot_pca_variance(pca)` | Bar + line kumulatif | Proses |
| Kurva keanggotaan vs fitur | `plot_membership_curve(X, U, feature)` | Line per cluster | Proses |
| Scatter cluster 2D | `plot_clusters(X2d, U, V2d, q_idx, rec_idx)` | Scatter, opacity = keanggotaan maks | Hasil |
| Keanggotaan film acuan | `plot_query_membership(u_q, names)` | Bar horizontal | Hasil |
| Radar pusat cluster | `plot_cluster_radar(V_original, names)` | Radar | Hasil |
| Precision@K | `plot_precision(df)` | Grouped bar fuzzy vs baseline | Evaluasi |
| Ukuran & silhouette per cluster | `plot_cluster_quality(...)` | Bar | Evaluasi |

### 6.3 Scatter cluster (grafik utama)
- Titik: warna = cluster dominan, opacity = 0.25 + 0.75 × keanggotaan maksimum.
- Hover: judul, tahun, genre, keanggotaan tiap cluster.
- Lapisan urutan: titik film → pusat cluster → film rekomendasi → film acuan (paling atas).
- Sumbu: "Komponen PCA 1" dan "Komponen PCA 2".

---

## 7. Isi Tiap Halaman

| Halaman | Isi wajib (urut dari atas) |
|---|---|
| **Beranda** (`app.py`) | Judul + tagline, deskripsi 2–3 kalimat, tujuan sistem, 3 kartu alur (Input → Proses → Hasil), tombol "Mulai", status data/model |
| **Input Data** | Pilihan sumber data (bawaan / unggah), uploader + hasil validasi, form input manual film baru, tabel penjelasan variabel, preview 10 baris data |
| **Proses** | Penjelasan tahapan (Steps 1–4), form parameter, tombol utama, metrik ringkas, grafik J_m, grafik pemilihan c, heatmap, expander rumus |
| **Hasil** | Pilih film acuan + k + α, tombol utama, kartu film acuan, interpretasi otomatis, tabel rekomendasi, scatter cluster, bar keanggotaan, radar cluster |
| **Evaluasi** | Penjelasan protokol uji, tombol utama, kartu metrik, grafik Precision@K, kualitas cluster, kelebihan & keterbatasan |
| **Informasi** | Penjelasan FCM & KNN, rumus, tabel parameter, diagram arsitektur, petunjuk penggunaan langkah demi langkah, info anggota |

---

## 8. Penulisan dan Istilah

- Bahasa Indonesia baku tapi ringan untuk teks GUI.
- Istilah tetap (jangan diganti-ganti):

| Pakai | Jangan |
|---|---|
| Derajat keanggotaan | Probabilitas, persentase cluster |
| Cluster | Kelompok, klaster (pilih satu: **cluster**) |
| Pusat cluster | Centroid (boleh di kurung) |
| Film acuan | Film input, film query |
| Skor kemiripan | Similarity score |
| KNN biasa | KNN standar, KNN non-fuzzy |

- Template interpretasi otomatis:
  > "**{judul}** paling kuat berada di cluster **{nama_cluster}** ({u_max:.2f}) dan juga memiliki keanggotaan {u_2:.2f} di cluster {nama_cluster_2}. Karena itu, rekomendasi didominasi film {genre_dominan} dengan rating rata-rata {rating_rata:.1f}."

---

## 9. State dan Alur Data di GUI

- Simpan data dan model di `st.session_state`: `df_film`, `X`, `artefak`, `fcm_result`, `last_reco`.
- Muat artefak dari `models/` dengan `@st.cache_resource` saat aplikasi dibuka.
- Fungsi berat (preprocessing, FCM) dibungkus `@st.cache_data`.
- Halaman yang butuh model mengecek dulu; jika belum ada, tampilkan `st.warning` + tombol menuju halaman Proses.

---

## 10. Checklist Konsistensi Sebelum Merge

- [ ] `st.set_page_config` sama di semua halaman
- [ ] Hanya satu tombol `type="primary"` per halaman
- [ ] Warna cluster dari `CLUSTER_COLORS`
- [ ] Semua input punya `help`
- [ ] Input tidak valid memunculkan `st.error` yang jelas
- [ ] Grafik dari `src/plots.py`, punya judul sumbu
- [ ] Ada expander "Bagaimana ini dihitung?" di halaman Proses, Hasil, Evaluasi
- [ ] Istilah sesuai tabel bagian 8
