"""Memuat data hasil preprocessing (Anggota 1) untuk KNN dan evaluasi."""
from cycler import V
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"


def _parse_genres(nilai):
    """Ubah string "[{'id': 28, 'name': 'Action'}, ...]" menjadi ['Action', ...]."""
    return [g["name"] for g in ast.literal_eval(nilai)]


def load_data(data_dir=DATA_DIR):
    """Mengembalikan X (n_film, n_fitur), G (n_film, n_genre), movies, dan daftar genre."""
    data_dir = Path(data_dir)
    X_df = pd.read_csv(data_dir / "X_final.csv", index_col=0)
    movies = pd.read_csv(data_dir / "movies_final.csv", index_col=0)

    if not movies.index.equals(X_df.index):
        raise ValueError("Urutan baris X_final dan movies_final tidak sama.")

    X = np.clip(X_df.to_numpy(dtype=float), 0.0, 1.0)
    if np.isnan(X).any():
        raise ValueError("X masih mengandung NaN.")

    daftar_genre = [_parse_genres(v) for v in movies["genres"]]
    genres = sorted({g for gs in daftar_genre for g in gs})
    kolom = {g: j for j, g in enumerate(genres)}

    G = np.zeros((len(movies), len(genres)), dtype=int)
    for i, gs in enumerate(daftar_genre):
        for g in gs:
            G[i, kolom[g]] = 1

    movies = movies.reset_index(drop=True)
    movies["genre_list"] = daftar_genre
    return X, G, movies, genres


def cari_film(movies, judul):
    """Mencari indeks baris film berdasarkan judul (tidak peka huruf besar/kecil)."""
    cocok = movies.index[movies["title"].str.lower() == judul.lower()]
    if len(cocok) == 0:
        raise ValueError(f"Film '{judul}' tidak ditemukan.")
    return int(cocok[0])

def load_model(n_film, c=3, model_dir=ROOT / "models", seed=42):
    """Muat U dan V hasil FCM; kalau belum ada, buat U acak sementara."""
    model_dir = Path(model_dir)
    u_path, v_path = model_dir / "U.npy", model_dir / "V.npy"
    if u_path.exists() and v_path.exists():
        U = np.load(u_path)
        V = np.load(v_path)
        if U.shape[0] != n_film or V.shape[0] != U.shape[1]:
            raise ValueError(
                f"Ukuran U {U.shape} / V {V.shape} tidak cocok dengan {n_film} film."
            )
        return U, V, True

    rng = np.random.default_rng(seed)
    U = rng.random((n_film, c))
    U /= U.sum(axis=1, keepdims=True)
    return U, None, False