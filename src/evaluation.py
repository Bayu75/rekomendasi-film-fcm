"""Evaluasi Precision@K untuk rekomendasi KNN berbobot fuzzy.

Sebuah film rekomendasi dianggap relevan jika berbagi setidaknya
`min_shared` genre dengan film kueri (default 1, artinya beririsan
minimal satu genre). Matriks genre G berbentuk one-hot/multi-hot
berukuran (n_film, n_genre), sesuai keluaran preprocessing.

    Precision@K = (jumlah rekomendasi yang relevan) / K
"""

import numpy as np
import pandas as pd
from src.data_loader import _parse_genres

from src.knn_fuzzy import recommend


def _cek_matriks_XUG(X, U, G):
    """Memastikan X, U, dan G adalah matriks 2 dimensi dengan jumlah film sama."""
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    G = np.asarray(G, dtype=float)
    if X.ndim != 2 or U.ndim != 2:
        raise ValueError("X dan U harus matriks 2 dimensi (n_film, n_fitur/n_cluster).")
    if G.ndim != 2:
        raise ValueError("G harus berupa matriks 2 dimensi berukuran (n_film, n_genre).")
    if not (X.shape[0] == U.shape[0] == G.shape[0]):
        raise ValueError(
            f"Jumlah film pada X ({X.shape[0]}), U ({U.shape[0]}), "
            f"dan G ({G.shape[0]}) harus sama."
        )
    if X.shape[0] == 0:
        raise ValueError("X, U, dan G tidak boleh kosong.")
    return X, U, G


def _cek_k(k, n):
    """k harus bilangan bulat antara 1 dan n-1."""
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not (1 <= k <= n - 1):
        raise ValueError(
            f"k harus berupa bilangan bulat antara 1 dan {n - 1}, diterima {k}."
        )
    return int(k)


def _pilih_kueri(n, query_indices, n_queries, seed):
    """Menentukan indeks film yang akan dievaluasi."""
    if query_indices is not None:
        q = np.asarray(query_indices, dtype=int).reshape(-1)
        if q.size == 0:
            raise ValueError("query_indices tidak boleh kosong.")
        if np.any((q < 0) | (q >= n)):
            raise ValueError(f"query_indices harus berada di antara 0 dan {n - 1}.")
        return q
    if n_queries is None:
        return np.arange(n)
    if (
        not isinstance(n_queries, (int, np.integer))
        or isinstance(n_queries, bool)
        or not (1 <= n_queries <= n)
    ):
        raise ValueError(
            f"n_queries harus berupa bilangan bulat antara 1 dan {n}, diterima {n_queries}."
        )
    return np.random.default_rng(seed).choice(n, size=int(n_queries), replace=False)


def precision_at_k(query_idx, rec_indices, G, min_shared=1):
    """Menghitung Precision@K untuk satu film kueri.

    Film di `rec_indices` relevan jika jumlah genre yang sama dengan
    film kueri minimal `min_shared` (default 1). K = panjang daftar
    rekomendasi. G adalah matriks one-hot/multi-hot (n_film, n_genre).
    """
    G = np.asarray(G, dtype=float)
    if G.ndim != 2:
        raise ValueError("G harus berupa matriks 2 dimensi berukuran (n_film, n_genre).")
    n = G.shape[0]
    if not isinstance(query_idx, (int, np.integer)) or isinstance(query_idx, bool):
        raise ValueError(f"query_idx harus berupa bilangan bulat, diterima {query_idx!r}.")
    query_idx = int(query_idx)
    if not (0 <= query_idx < n):
        raise ValueError(
            f"query_idx harus berada di antara 0 dan {n - 1}, diterima {query_idx}."
        )
    rec_indices = np.asarray(rec_indices, dtype=int).reshape(-1)
    if rec_indices.size == 0:
        raise ValueError("Daftar rekomendasi rec_indices tidak boleh kosong.")
    if np.any((rec_indices < 0) | (rec_indices >= n)):
        raise ValueError(f"Indeks rekomendasi harus berada di antara 0 dan {n - 1}.")
    if not isinstance(min_shared, (int, np.integer)) or isinstance(min_shared, bool) or min_shared < 1:
        raise ValueError(f"min_shared harus berupa bilangan bulat ≥ 1, diterima {min_shared}.")

    # Irisan genre = hasil kali titik dua vektor one-hot.
    n_relevan = np.sum((G[rec_indices] @ G[query_idx]) >= min_shared)
    return float(n_relevan / rec_indices.size)


def mean_precision_at_k(
    X, U, G, k=10, alpha=0.5, query_indices=None, n_queries=None, seed=42, min_shared=1
):
    """Rata-rata Precision@K pada sejumlah film kueri.

    Jika `query_indices` tidak diisi, semua film dipakai, atau sebanyak
    `n_queries` film dipilih acak dengan `seed`.
    """
    X, U, G = _cek_matriks_XUG(X, U, G)
    k = _cek_k(k, X.shape[0])
    kueri = _pilih_kueri(X.shape[0], query_indices, n_queries, seed)
    skor = [
        precision_at_k(int(q), recommend(X, U, int(q), k=k, alpha=alpha)[0], G, min_shared)
        for q in kueri
    ]
    return float(np.mean(skor))


def compare_methods(
    X, U, G, k=10, alpha=0.5, query_indices=None, n_queries=None, seed=42, min_shared=1
):
    """Membandingkan KNN fuzzy dengan KNN biasa (α = 1) pada kueri yang sama.

    Mengembalikan dictionary: precision_fuzzy, precision_baseline,
    selisih (fuzzy − baseline; positif berarti fuzzy lebih baik),
    plus alpha dan k yang dipakai.
    """
    X, U, G = _cek_matriks_XUG(X, U, G)
    k = _cek_k(k, X.shape[0])
    kueri = _pilih_kueri(X.shape[0], query_indices, n_queries, seed)
    kwargs = dict(X=X, U=U, G=G, k=k, query_indices=kueri, min_shared=min_shared)
    p_fuzzy = mean_precision_at_k(alpha=alpha, **kwargs)
    p_baseline = mean_precision_at_k(alpha=1.0, **kwargs)
    return {
        "precision_fuzzy": p_fuzzy,
        "precision_baseline": p_baseline,
        "selisih": p_fuzzy - p_baseline,
        "alpha": float(alpha),
        "k": int(k),
    }


def alpha_sweep(
    X, U, G, alphas=(0.0, 0.25, 0.5, 0.75, 1.0), k=10,
    query_indices=None, n_queries=None, seed=42, min_shared=1,
):
    """Menghitung Precision@K untuk sejumlah nilai α, siap untuk grafik BAB IV.

    Mengembalikan list dictionary {"alpha": ..., "precision": ...}.
    Semua α memakai himpunan kueri yang sama agar perbandingannya adil.
    """
    X, U, G = _cek_matriks_XUG(X, U, G)
    k = _cek_k(k, X.shape[0])
    alphas = list(alphas)
    if len(alphas) == 0:
        raise ValueError("Daftar alphas tidak boleh kosong.")
    for a in alphas:
        if not (0.0 <= float(a) <= 1.0):
            raise ValueError(f"Setiap alpha harus berada di antara 0 dan 1, diterima {a}.")
    kueri = _pilih_kueri(X.shape[0], query_indices, n_queries, seed)
    return [
        {
            "alpha": float(a),
            "precision": mean_precision_at_k(
                X, U, G, k=k, alpha=float(a), query_indices=kueri, min_shared=min_shared
            ),
        }
        for a in alphas
    ]

def _genre_matrix(df_film):
    """Bentuk matriks G (one-hot genre) dari kolom genres."""
    daftar = [g if isinstance(g, list) else _parse_genres(g) for g in df_film["genres"]]
    semua = sorted({nama for gs in daftar for nama in gs})
    return np.array([[nama in gs for nama in semua] for gs in daftar], dtype=int)


def precision_at_k_prd(X, U, df_film, k_list=(5, 10, 20), n_sample=200, alpha=0.5, seed=42):
    """Versi sesuai PRD untuk GUI. Hasil: DataFrame k, precision_fuzzy, precision_baseline."""
    G = _genre_matrix(df_film)
    n_sample = min(int(n_sample), len(X))
    baris = []
    for k in k_list:
        h = compare_methods(X, U, G, k=k, alpha=alpha, n_queries=n_sample, seed=seed)
        baris.append({"k": int(k),
                      "precision_fuzzy": h["precision_fuzzy"],
                      "precision_baseline": h["precision_baseline"]})
    return pd.DataFrame(baris)