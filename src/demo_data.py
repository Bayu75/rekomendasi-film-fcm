"""DATA DEMO / PENGGANTI SEMENTARA untuk GUI FuzzyFlix.

Modul ini HANYA dipakai bila modul asli (preprocessing, fcm, knn_fuzzy,
evaluation) belum tersedia atau masih berupa stub. Semua fungsi publik meniru
*nama dan signature* kontrak antarmodul PRD bagian 8, sehingga
``src/provider.py`` bisa menukar demo ↔ asli tanpa mengubah kode halaman.

PERINGATAN: ini bukan implementasi resmi anggota 1–3. Film di sini sintetis
(judul karangan), dan algoritma dibuat sesingkat mungkin hanya agar semua halaman
bisa dirender dan dites. Jangan dipakai untuk laporan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from src.interpret import KOLOM_WAJIB, parse_genres

GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
          "Family", "Fantasy", "Foreign", "History", "Horror", "Music", "Mystery",
          "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"]

NUM_FEATURES = ["vote_average", "vote_count", "popularity", "runtime", "year"]
FEATURE_NAMES = ["Rating", "Jumlah vote", "Popularitas", "Durasi", "Tahun rilis"]

# (peluang genre, rating, log-vote, log-popularitas, durasi, tahun, sebaran tahun)
_ARKETIPE = [
    ({"Drama": .9, "Romance": .5, "History": .2}, 7.3, 5.6, 2.2, 125, 1975, 12),
    ({"Action": .9, "Adventure": .6, "Science Fiction": .4, "Thriller": .3}, 6.4, 7.8, 3.2, 125, 2008, 5),
    ({"Comedy": .9, "Family": .5, "Animation": .35, "Fantasy": .2}, 6.3, 6.5, 2.8, 98, 2003, 8),
    ({"Horror": .8, "Thriller": .7, "Mystery": .3}, 5.7, 5.9, 2.6, 100, 2005, 8),
    ({"Crime": .8, "Drama": .5, "Thriller": .4, "War": .15}, 7.0, 6.8, 2.9, 115, 1995, 10),
]
_KATA_1 = ["Bayangan", "Senja", "Samudra", "Kota", "Rahasia", "Cahaya", "Malam", "Jejak", "Api",
           "Badai", "Mimpi", "Perjalanan", "Gerbang", "Lorong", "Pulau", "Dendam", "Harapan",
           "Langit", "Janji", "Petualangan", "Musim", "Warisan", "Fajar", "Gema"]
_KATA_2 = ["Terakhir", "Biru", "Tersembunyi", "Merah", "Sunyi", "Abadi", "Gelap", "Liar", "Pertama",
           "Hilang", "Emas", "Jauh", "Rapuh", "Berdarah", "Ajaib", "Dingin", "Lama", "Baru"]


# ---------------------------------------------------------------------------
# Data sintetis
# ---------------------------------------------------------------------------

def buat_data_mentah(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Buat tabel film sintetis berkolom wajib CSV (PRD bagian 4).

    5 arketipe film + ±15% film campuran dua arketipe, sehingga ada film dengan
    keanggotaan terbagi di beberapa cluster.
    """
    rng = np.random.default_rng(seed)
    baris, judul_dipakai = [], set()
    for i in range(n):
        a = int(rng.integers(len(_ARKETIPE)))
        prm = list(_ARKETIPE[a])
        peluang = dict(prm[0])
        if rng.random() < 0.15:  # film campuran
            b = int(rng.integers(len(_ARKETIPE)))
            pb = _ARKETIPE[b]
            for g, p in pb[0].items():
                peluang[g] = max(peluang.get(g, 0), p * 0.8)
            prm[1:] = [(x + y) / 2 for x, y in zip(prm[1:], pb[1:])]
        genre = [g for g in GENRES if rng.random() < peluang.get(g, 0.01)]
        if not genre:
            genre = [max(peluang, key=peluang.get)]
        rating = float(np.clip(rng.normal(prm[1], 0.4), 1.5, 9.4))
        votes = float(np.clip(np.expm1(rng.normal(prm[2], 0.5)), 50, 14000))
        pop = float(np.clip(np.expm1(rng.normal(prm[3], 0.4)), 0.1, 500))
        runtime = float(np.clip(rng.normal(prm[4], 9), 60, 240))
        tahun = int(np.clip(rng.normal(prm[5], prm[6] * 0.6), 1930, 2017))
        for _ in range(20):
            t = f"{_KATA_1[rng.integers(len(_KATA_1))]} {_KATA_2[rng.integers(len(_KATA_2))]}"
            if t not in judul_dipakai:
                break
        else:
            t = f"{t} {i}"
        judul_dipakai.add(t)
        baris.append({"id": 900001 + i, "title": t, "vote_average": round(rating, 1),
                      "vote_count": int(votes), "popularity": round(pop, 3), "runtime": int(runtime),
                      "release_date": f"{tahun}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}",
                      "genres": str([{"id": GENRES.index(g), "name": g} for g in genre]),
                      "poster_path": None,  # opsional: demo memakai poster pengganti (tanpa jaringan)
                      "overview": (f"Film bergenre {', '.join(g.lower() for g in genre[:2])} berjudul \"{t}\". "
                                   "Sinopsis ini karangan untuk mode demo; isi cerita tidak dipakai dalam perhitungan.")})
    return pd.DataFrame(baris, columns=KOLOM_WAJIB + ["poster_path", "overview"])


# ---------------------------------------------------------------------------
# Pengganti preprocessing.py (kontrak PRD bagian 8)
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    """Baca CSV, ubah tipe, buang duplikat id, filter ringan (versi demo)."""
    df = pd.read_csv(path, low_memory=False)
    for k in ("vote_average", "vote_count", "popularity", "runtime"):
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["vote_average", "vote_count", "popularity", "runtime", "release_date", "title"])
    df = df[(df["vote_count"] >= 50) & (df["runtime"] > 0)]
    df = df[df["genres"].apply(lambda g: len(parse_genres(g)) > 0)]
    return df.drop_duplicates(subset="id").reset_index(drop=True)


def _numerik(df: pd.DataFrame) -> pd.DataFrame:
    tahun = pd.to_datetime(df["release_date"], errors="coerce").dt.year.astype(float)
    return pd.DataFrame({"vote_average": pd.to_numeric(df["vote_average"], errors="coerce"),
                         "vote_count": np.log1p(pd.to_numeric(df["vote_count"], errors="coerce")),
                         "popularity": np.log1p(pd.to_numeric(df["popularity"], errors="coerce")),
                         "runtime": pd.to_numeric(df["runtime"], errors="coerce"),
                         "year": tahun})


def _onehot(genre_list) -> np.ndarray:
    M = np.zeros((len(genre_list), len(GENRES)))
    for i, gs in enumerate(genre_list):
        for g in gs:
            if g in GENRES:
                M[i, GENRES.index(g)] = 1
    return M


def build_features(df: pd.DataFrame):
    """Bangun X, df_film, artefak (versi demo): min-max numerik + PCA genre."""
    df = df.reset_index(drop=True)
    genre_list = df["genres"].apply(parse_genres).tolist()
    num = _numerik(df)
    scaler = MinMaxScaler().fit(num.to_numpy())
    pca = PCA(n_components=3, random_state=42).fit(_onehot(genre_list))
    X = np.hstack([scaler.transform(num.to_numpy()), pca.transform(_onehot(genre_list))])
    pca2 = PCA(n_components=2, random_state=42).fit(X)
    df_film = pd.DataFrame({"id": df["id"], "title": df["title"].astype(str), "genres": genre_list,
                            "vote_average": num["vote_average"], "vote_count": pd.to_numeric(df["vote_count"]),
                            "popularity": pd.to_numeric(df["popularity"]), "runtime": num["runtime"],
                            "year": num["year"].astype(int)})
    artefak = {"scaler": scaler, "pca": pca, "pca_2d": pca2,
               "feature_names": FEATURE_NAMES + [f"Genre PC{i + 1}" for i in range(pca.n_components_)],
               "demo": True}
    return X, df_film, artefak


def validate_csv(df: pd.DataFrame) -> list[str]:
    """Validasi CSV (versi demo) -> daftar pesan error."""
    from src.interpret import validasi_csv_cadangan
    return validasi_csv_cadangan(df)


def transform_new(film: dict, artefak: dict) -> np.ndarray:
    """Ubah dict film baru menjadi vektor fitur x (versi demo).

    Kunci dict: vote_average, vote_count, popularity, runtime, year, genres.
    """
    num = np.array([[film["vote_average"], np.log1p(film["vote_count"]), np.log1p(film["popularity"]),
                     film["runtime"], film["year"]]], dtype=float)
    xn = np.clip(artefak["scaler"].transform(num), 0, 1)
    xg = artefak["pca"].transform(_onehot([parse_genres(film["genres"])]))
    return np.hstack([xn, xg])[0]


# ---------------------------------------------------------------------------
# Pengganti fcm.py
# ---------------------------------------------------------------------------

def _jarak(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    d = np.sqrt(((X[:, None, :] - V[None, :, :]) ** 2).sum(axis=2))
    return np.fmax(d, 1e-12)


def _u_dari_jarak(d: np.ndarray, m: float) -> np.ndarray:
    p = 2.0 / (m - 1.0)
    return 1.0 / ((d[:, :, None] / d[:, None, :]) ** p).sum(axis=2)


def fcm(X: np.ndarray, c: int, m: float = 2.0, error: float = 1e-5,
        maxiter: int = 300, seed: int = 42) -> dict:
    """Fuzzy C-Means ringkas (versi demo); format keluaran sama dengan kontrak."""
    rng = np.random.default_rng(seed)
    # inisialisasi: c film acak sebagai pusat awal (lebih stabil daripada U acak)
    V = X[rng.choice(X.shape[0], size=c, replace=False)]
    U = _u_dari_jarak(_jarak(X, V), m)
    J_hist, n_iter = [], 0
    for n_iter in range(1, maxiter + 1):
        Um = U ** m
        V = (Um.T @ X) / Um.sum(axis=0)[:, None]
        d = _jarak(X, V)
        U_baru = _u_dari_jarak(d, m)
        J_hist.append(float(((U_baru ** m) * d ** 2).sum()))
        selesai = np.abs(U_baru - U).max() < error
        U = U_baru
        if selesai:
            break
    Um = U ** m
    V = (Um.T @ X) / Um.sum(axis=0)[:, None]
    return {"V": V, "U": U, "J_history": J_hist, "n_iter": n_iter,
            "fpc": float((U ** 2).sum() / X.shape[0])}


def membership(x: np.ndarray, V: np.ndarray, m: float = 2.0) -> np.ndarray:
    """Derajat keanggotaan satu film terhadap pusat cluster V."""
    d = _jarak(np.atleast_2d(x), V)
    return _u_dari_jarak(d, m)[0]


def select_c(X: np.ndarray, c_range=range(3, 9), **kw) -> pd.DataFrame:
    """FPC dan silhouette untuk tiap c (versi demo). Kolom: c, fpc, silhouette."""
    baris = []
    for c in c_range:
        r = fcm(X, int(c), **kw)
        lab = r["U"].argmax(axis=1)
        sil = float(silhouette_score(X, lab, sample_size=min(2000, len(X)), random_state=42)) \
            if len(set(lab)) > 1 else float("nan")
        baris.append({"c": int(c), "fpc": r["fpc"], "silhouette": sil})
    return pd.DataFrame(baris)


# ---------------------------------------------------------------------------
# Pengganti knn_fuzzy.py dan evaluation.py
# ---------------------------------------------------------------------------

def recommend(q_idx, X, U, k: int = 10, alpha: float = 0.5, x_new=None, u_new=None) -> pd.DataFrame:
    """KNN berbobot fuzzy (versi demo): d = α‖x_q−x_i‖ + (1−α)‖u_q−u_i‖; skor = 1/(1+d)."""
    if q_idx is not None:
        xq, uq = X[q_idx], U[q_idx]
    else:
        xq, uq = np.asarray(x_new, dtype=float), np.asarray(u_new, dtype=float)
    d = alpha * np.linalg.norm(X - xq, axis=1) + (1 - alpha) * np.linalg.norm(U - uq, axis=1)
    if q_idx is not None:
        d[q_idx] = np.inf
    idx = np.argsort(d)[:k]
    return pd.DataFrame({"idx": idx, "distance": d[idx], "score": 1.0 / (1.0 + d[idx])})


def precision_at_k(X, U, df_film, k_list=(5, 10, 20), n_sample=200, alpha=0.5, seed=42) -> pd.DataFrame:
    """Precision@K fuzzy vs KNN biasa; relevan = berbagi ≥ 1 genre (versi demo)."""
    rng = np.random.default_rng(seed)
    q = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    genre = [set(parse_genres(g)) for g in df_film["genres"]]
    kmax = max(k_list)
    hasil = {a: {k: [] for k in k_list} for a in ("fuzzy", "base")}
    for qi in q:
        for nama, a in (("fuzzy", alpha), ("base", 1.0)):
            idx = recommend(int(qi), X, U, k=kmax, alpha=a)["idx"].to_numpy()
            rel = np.array([len(genre[qi] & genre[i]) > 0 for i in idx], dtype=float)
            for k in k_list:
                hasil[nama][k].append(rel[:k].mean())
    return pd.DataFrame({"k": list(k_list),
                         "precision_fuzzy": [float(np.mean(hasil["fuzzy"][k])) for k in k_list],
                         "precision_baseline": [float(np.mean(hasil["base"][k])) for k in k_list]})
