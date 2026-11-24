# FuzzyFlix

Sistem rekomendasi film berbasis **Fuzzy C-Means (FCM)** dan **K-Nearest Neighbors (KNN)** berbobot derajat keanggotaan.
Final Project mata kuliah Teknologi Logika Fuzzy, Universitas Udayana.

## Dataset

[The Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).
Unduh lalu taruh `movies_metadata.csv` (dan opsional `ratings_small.csv`) di folder `data/raw/TheMovieDataset`.
File dataset tidak di-commit ke Git.

## Cara menjalankan

```bash
git clone <url-repo>
cd fuzzyflix
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Struktur folder

```text
rekomendasi-film-fcm/
├── app.py                    # halaman Beranda
├── pages/
│   ├── 1_Input_Data.py       # input manual dan unggah CSV
│   ├── 2_Proses.py           # menjalankan FCM
│   ├── 3_Hasil.py            # rekomendasi KNN berbobot fuzzy
│   ├── 4_Evaluasi.py         # Precision@K dan perbandingan metode
│   └── 5_Informasi.py        # penjelasan metode dan petunjuk
├── src/
│   ├── preprocessing.py      # pembersihan data, fitur, validasi CSV
│   ├── data_loader.py        # memuat data dan model
│   ├── fcm.py                # Fuzzy C-Means
│   ├── knn_fuzzy.py          # KNN dengan jarak gabungan
│   ├── evaluation.py         # Precision@K
│   ├── provider.py           # penghubung GUI dengan modul algoritma
│   ├── interpret.py          # interpretasi hasil
│   ├── contoh.py             # contoh perhitungan sederhana
│   ├── plots.py              # grafik Plotly
│   ├── komponen.py, ui.py, tema.py   # komponen dan tema GUI
│   └── demo_data.py          # data cadangan mode demo
├── data/
│   ├── raw/                  # dataset Kaggle (tidak di-commit)
│   └── processed/            # X_final.csv, movies_final.csv, G.csv, preprocessing_artifacts.pkl
├── models/                   # U.npy, V.npy, pca_fcm.joblib, fcm_meta.json
├── notebooks/                # eksplorasi data, pembuatan model FCM, eksperimen evaluasi
├── reports/                  # hasil eksperimen (perbandingan metode, alpha sweep)
├── tests/                    # unit test dan smoke test GUI (pytest)
├── docs/                     # PRD.md, design-system.md, GUI-notes.md
├── requirements.txt
└── pytest.ini
```

## Kontribusi anggota

| No  | Nama/NIM                                       | Tanggung jawab           | Kontribusi konkret                                                         | Bukti pekerjaan                                            |
| --- | ---------------------------------------------- | ------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 1   | Tebuana Agung Ayudia Lory (2408561080)         | Data dan preprocessing   | Pembersihan data, fitur, PCA genre, validasi CSV                           | `src/preprocessing.py`, BAB I dan III                      |
| 2   | Gede Widhi Wardhana (2308561038)               | Fuzzy C-Means            | Implementasi FCM, pemilihan c, contoh hitung FCM                           | `src/fcm.py`, BAB II dan IV                                |
| 3   | Kadek Bayu Dwi Pradnya (2408661059)            | KNN, integrasi, evaluasi | KNN berbobot fuzzy, Precision@K, contoh hitung KNN                         | `src/knn_fuzzy.py`, `src/evaluation.py`, BAB II dan BAB IV |
| 4   | Ni Luh Putu Ayu Maytta Wulandhani (2408561061) | GUI dan visualisasi      | Enam halaman, grafik Plotly, interpretasi, contoh perhitungan, pesan error | `app.py`, `pages/`, `src/plots.py`, BAB III dan BAB IV     |

## Alur kerja Git

1. Selalu `git pull origin main` sebelum mulai.
2. Kerjakan di branch masing-masing: `git checkout -b <branch>`.
3. Commit kecil dengan pesan jelas, misalnya `feat(fcm): update derajat keanggotaan`.
4. Push lalu buat Pull Request ke `main`, minta 1 anggota lain review.
5. Jangan push langsung ke `main`.

Detail kebutuhan sistem ada di `docs/PRD.md`, aturan tampilan di `docs/design-system.md`.
