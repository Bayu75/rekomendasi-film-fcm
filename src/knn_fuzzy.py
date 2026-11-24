"""KNN berbobot keanggotaan fuzzy untuk rekomendasi film.

Kemiripan dua film diukur dari dua hal sekaligus:

    d = α * ||x_q - x_i|| + (1 - α) * ||u_q - u_i||

Bagian pertama membandingkan fitur, bagian kedua membandingkan
derajat keanggotaan cluster FCM. α mengatur bobot keduanya
(default 0,5). Jika α = 1, hasilnya sama dengan KNN biasa.
"""

import numpy as np


def _cek_alpha(alpha):
    """Memastikan alpha berada di rentang [0, 1]."""
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(
            f"alpha harus berada di antara 0 dan 1 (inklusif), diterima {alpha}."
        )
    return float(alpha)


def _cek_matriks_XU(X, U):
    """Mengubah X dan U menjadi array dan memeriksa bentuknya."""
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    if X.ndim != 2 or U.ndim != 2:
        raise ValueError("X dan U harus matriks 2 dimensi (n_film, n_fitur/n_cluster).")
    if X.shape[0] != U.shape[0]:
        raise ValueError(
            f"Jumlah film pada X ({X.shape[0]}) dan U ({U.shape[0]}) harus sama."
        )
    if X.shape[0] == 0:
        raise ValueError("X dan U tidak boleh kosong.")
    return X, U


def combined_distance(X, U, q_x, q_u, alpha=0.5):
    """Menghitung jarak gabungan dari film kueri ke semua film di dataset.

    X berukuran (n_film, n_fitur), U berukuran (n_film, n_cluster).
    q_x dan q_u adalah vektor fitur dan keanggotaan film kueri.
    Mengembalikan array jarak berukuran (n_film,).
    """
    X, U = _cek_matriks_XU(X, U)
    alpha = _cek_alpha(alpha)
    q_x = np.asarray(q_x, dtype=float).reshape(-1)
    q_u = np.asarray(q_u, dtype=float).reshape(-1)
    if q_x.shape[0] != X.shape[1]:
        raise ValueError(
            f"Panjang q_x ({q_x.shape[0]}) harus sama dengan jumlah fitur X ({X.shape[1]})."
        )
    if q_u.shape[0] != U.shape[1]:
        raise ValueError(
            f"Panjang q_u ({q_u.shape[0]}) harus sama dengan jumlah cluster U ({U.shape[1]})."
        )

    # Jarak Euclidean per baris: akar jumlah kuadrat selisih.
    jarak_fitur = np.linalg.norm(X - q_x, axis=1)
    jarak_keanggotaan = np.linalg.norm(U - q_u, axis=1)
    return alpha * jarak_fitur + (1.0 - alpha) * jarak_keanggotaan


def recommend(X, U, query_idx, k=10, alpha=0.5):
    """Mengembalikan k film terdekat dari film pada indeks query_idx.

    Film kueri sendiri dikeluarkan dari hasil. Mengembalikan
    (indeks, jarak) masing-masing berukuran (k,), terurut dari yang terdekat.
    """
    X, U = _cek_matriks_XU(X, U)
    alpha = _cek_alpha(alpha)
    n = X.shape[0]
    if n < 2:
        raise ValueError("Diperlukan minimal 2 film untuk membuat rekomendasi.")
    if not isinstance(query_idx, (int, np.integer)) or isinstance(query_idx, bool):
        raise ValueError(f"query_idx harus berupa bilangan bulat, diterima {query_idx!r}.")
    query_idx = int(query_idx)
    if not (0 <= query_idx < n):
        raise ValueError(
            f"query_idx harus berada di antara 0 dan {n - 1}, diterima {query_idx}."
        )
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not (1 <= k <= n - 1):
        raise ValueError(
            f"k harus berupa bilangan bulat antara 1 dan {n - 1}, diterima {k}."
        )

    jarak = combined_distance(X, U, X[query_idx], U[query_idx], alpha=alpha)
    jarak = jarak.copy()
    jarak[query_idx] = np.inf  # keluarkan film kueri dari kandidat
    indeks = np.argsort(jarak)[: int(k)]
    return indeks, jarak[indeks]


def membership_new(x_new, V, m=2.0):
    """Menghitung keanggotaan film baru terhadap pusat cluster FCM.

    Rumusnya sama dengan pembaruan U pada FCM:

        u_j = 1 / Σ_k ( ||x - v_j|| / ||x - v_k|| ) ^ (2 / (m - 1))

    Jika film tepat di salah satu pusat (jarak nol), hasilnya one-hot:
    1 pada cluster itu dan 0 pada cluster lain. V berada di ruang fitur
    yang sama dengan X. Keluaran berukuran (n_cluster,) dan jumlahnya 1.
    """
    x_new = np.asarray(x_new, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] == 0:
        raise ValueError("V harus matriks 2 dimensi berukuran (n_cluster, n_fitur).")
    if x_new.shape[0] != V.shape[1]:
        raise ValueError(
            f"Panjang x_new ({x_new.shape[0]}) harus sama dengan jumlah fitur V ({V.shape[1]})."
        )
    m = float(m)
    if m <= 1.0:
        raise ValueError(f"Parameter fuzzifier m harus lebih besar dari 1, diterima {m}.")

    jarak = np.linalg.norm(V - x_new, axis=1)
    tepat_di_pusat = np.isclose(jarak, 0.0)
    if np.any(tepat_di_pusat):
        u = np.zeros(V.shape[0], dtype=float)
        # Jika tepat di lebih dari satu pusat, keanggotaan dibagi rata.
        u[tepat_di_pusat] = 1.0 / np.sum(tepat_di_pusat)
        return u

    # Bentuk setara yang lebih stabil: u_j = d_j^(-2/(m-1)) / Σ_k d_k^(-2/(m-1))
    bobot = jarak ** (-2.0 / (m - 1.0))
    return bobot / bobot.sum()


def recommend_new(x_new, X, U, V, k=10, alpha=0.5, m=2.0):
    """Merekomendasikan k film terdekat untuk film yang belum ada di dataset.

    Keanggotaan film baru dihitung dari pusat V, lalu semua film di X
    diurutkan dengan jarak gabungan. Tidak ada film yang dikeluarkan
    karena kueri belum ada di dataset. Mengembalikan (indeks, jarak).
    """
    X, U = _cek_matriks_XU(X, U)
    alpha = _cek_alpha(alpha)
    n = X.shape[0]
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not (1 <= k <= n):
        raise ValueError(
            f"k harus berupa bilangan bulat antara 1 dan {n}, diterima {k}."
        )

    u_new = membership_new(x_new, V, m=m)
    jarak = combined_distance(X, U, x_new, u_new, alpha=alpha)
    indeks = np.argsort(jarak)[: int(k)]
    return indeks, jarak[indeks]
