# FuzzyFlix

Sistem rekomendasi film berbasis **Fuzzy C-Means (FCM)** dan **K-Nearest Neighbors (KNN)** berbobot derajat keanggotaan.
Final Project mata kuliah Teknologi Logika Fuzzy, Universitas Udayana.

## Dataset

[The Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).
Unduh lalu taruh `movies_metadata.csv` (dan opsional `ratings_small.csv`) di folder `data/raw/`.
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
fuzzyflix/
├── app.py              # halaman Beranda
├── pages/              # Input Data, Proses, Hasil, Evaluasi, Informasi Sistem
├── src/
│   ├── preprocessing.py
│   ├── fcm.py
│   ├── knn_fuzzy.py
│   ├── evaluation.py
│   └── plots.py
├── data/raw/           # dataset Kaggle (tidak di-commit)
├── models/             # hasil training (tidak di-commit)
├── notebooks/          # eksplorasi data
├── tests/
└── docs/               # PRD.md, design-system.md
```

## Pembagian tugas

| Anggota | Peran | File utama | Branch |
|---|---|---|---|
| Anggota 1 | Data & preprocessing | `src/preprocessing.py` | `data` |
| Anggota 2 | Fuzzy C-Means | `src/fcm.py` | `fcm` |
| Anggota 3 | KNN & evaluasi | `src/knn_fuzzy.py`, `src/evaluation.py` | `knn-eval` |
| Anggota 4 | GUI & visualisasi | `app.py`, `pages/`, `src/plots.py` | `gui` |

## Alur kerja Git

1. Selalu `git pull origin main` sebelum mulai.
2. Kerjakan di branch masing-masing: `git checkout -b <branch>`.
3. Commit kecil dengan pesan jelas, misalnya `feat(fcm): update derajat keanggotaan`.
4. Push lalu buat Pull Request ke `main`, minta 1 anggota lain review.
5. Jangan push langsung ke `main`.

Detail kebutuhan sistem ada di `docs/PRD.md`, aturan tampilan di `docs/design-system.md`.
