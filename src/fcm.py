"""
Modul Fuzzy C-Means (FCM) untuk FuzzyFlix.

Ditulis sendiri dengan NumPy (tanpa skfuzzy) supaya setiap langkah algoritma
dapat dijelaskan. Hasilnya dicocokkan dengan ``skfuzzy.cmeans`` pada
``experiments/compare_skfuzzy.py`` dan ``tests/test_fcm.py``.

Notasi yang dipakai (sama dengan laporan):
    n  = jumlah data (film), f = jumlah fitur, c = jumlah cluster
    X  = matriks data, bentuk (n, f)
    U  = matriks keanggotaan, bentuk (n, c); U[i, j] = u_ij = derajat
         keanggotaan data ke-i pada cluster ke-j
    V  = matriks pusat cluster, bentuk (c, f); V[j] = v_j
    m  = pangkat fuzzifier (m > 1), umumnya m = 2
    d_ij = jarak Euclidean antara x_i dan v_j

Rumus:
    v_j  = sum_i (u_ij^m * x_i) / sum_i (u_ij^m)
    u_ij = 1 / sum_k (d_ij / d_ik)^(2/(m-1))
    J_m  = sum_i sum_j u_ij^m * d_ij^2
    FPC  = (1/n) * sum_i sum_j u_ij^2
    Berhenti jika max|U_baru - U_lama| < epsilon atau iterasi = maxiter.

Catatan: skfuzzy memakai U berbentuk (c, n), sedangkan modul ini memakai
(n, c) seperti kontrak PRD. Saat membandingkan, U dari skfuzzy di-transpose.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

__all__ = [
    "fcm",
    "membership",
    "select_c",
    "init_membership",
    "update_centers",
    "distances",
    "update_membership",
    "objective",
    "fpc",
    "dominant_cluster",
]


# ---------------------------------------------------------------------------
# Validasi masukan
# ---------------------------------------------------------------------------
def _validasi_X(X) -> np.ndarray:
    """Memastikan X adalah matriks 2 dimensi berisi angka berhingga.

    Mengembalikan X dalam bentuk np.ndarray bertipe float64.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("X harus berisi angka (numerik).") from None
    if X.ndim != 2:
        raise ValueError(
            f"X harus berupa matriks 2 dimensi (n_data, n_fitur), "
            f"tetapi yang diberikan berdimensi {X.ndim}."
        )
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X tidak boleh kosong (minimal 1 data dan 1 fitur).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X tidak boleh mengandung NaN atau nilai tak hingga (inf).")
    return X


def _validasi_m(m) -> float:
    """Memastikan pangkat fuzzifier m adalah angka lebih besar dari 1."""
    if isinstance(m, bool) or not isinstance(m, numbers.Real) or not np.isfinite(m):
        raise ValueError("m (fuzzifier) harus berupa angka berhingga.")
    if m <= 1:
        raise ValueError(f"m (fuzzifier) harus lebih besar dari 1, diberikan m={m}.")
    return float(m)


def _validasi_c(c, n: int) -> int:
    """Memastikan jumlah cluster c adalah bilangan bulat dengan 2 <= c <= n."""
    if isinstance(c, bool) or not isinstance(c, numbers.Integral):
        raise ValueError(f"c (jumlah cluster) harus bilangan bulat, diberikan {c!r}.")
    if c < 2:
        raise ValueError(f"c (jumlah cluster) minimal 2, diberikan c={c}.")
    if c > n:
        raise ValueError(
            f"c (jumlah cluster) tidak boleh melebihi jumlah data n={n}, diberikan c={c}."
        )
    return int(c)


# ---------------------------------------------------------------------------
# Fungsi-fungsi pembantu (satu fungsi = satu langkah algoritma)
# ---------------------------------------------------------------------------
def init_membership(n: int, c: int, seed: int = 42) -> np.ndarray:
    """Langkah 1: membuat matriks keanggotaan awal U secara acak.

    Setiap baris U (satu data) dinormalisasi sehingga jumlahnya 1, karena
    syarat FCM adalah sum_j u_ij = 1 untuk setiap data i.

    Parameter seed membuat hasil acak dapat diulang (reproducible).
    Mengembalikan U berbentuk (n, c).
    """
    # Generator acak modern NumPy; seed yang sama -> bilangan acak yang sama.
    rng = np.random.default_rng(seed)
    # Bilangan acak di (0, 1) untuk setiap pasangan (data, cluster).
    U = rng.random((n, c))
    # Hindari nilai nol persis agar setiap data punya keanggotaan > 0.
    U = np.fmax(U, np.finfo(np.float64).eps)
    # Normalisasi per baris: bagi setiap elemen dengan jumlah barisnya.
    U = U / U.sum(axis=1, keepdims=True)
    return U


def update_centers(X: np.ndarray, U: np.ndarray, m: float = 2.0) -> np.ndarray:
    """Langkah 2: menghitung pusat cluster V dari U.

    Rumus: v_j = sum_i (u_ij^m * x_i) / sum_i (u_ij^m)
    Artinya pusat cluster adalah rata-rata berbobot seluruh data, dengan
    bobot u_ij^m. Data yang keanggotaannya besar lebih menarik pusat.
    Mengembalikan V berbentuk (c, f).
    """
    # Bobot setiap data untuk setiap cluster: u_ij^m, bentuk (n, c).
    Um = U ** m
    # Pembilang: sum_i u_ij^m * x_i untuk setiap cluster j -> (c, f).
    pembilang = Um.T @ X
    # Penyebut: sum_i u_ij^m untuk setiap cluster j -> (c, 1).
    penyebut = Um.sum(axis=0)[:, None]
    return pembilang / penyebut


def distances(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Menghitung jarak Euclidean d_ij antara setiap data x_i dan pusat v_j.

    d_ij = sqrt( sum_f (x_if - v_jf)^2 )
    Mengembalikan D berbentuk (n, c).
    """
    # Selisih setiap data terhadap setiap pusat memakai broadcasting:
    # X[:, None, :] berbentuk (n, 1, f), V[None, :, :] berbentuk (1, c, f).
    selisih = X[:, None, :] - V[None, :, :]
    # Jumlahkan kuadrat selisih pada sumbu fitur, lalu akar.
    return np.sqrt(np.sum(selisih ** 2, axis=2))


def update_membership(D: np.ndarray, m: float = 2.0) -> np.ndarray:
    """Langkah 3: menghitung keanggotaan baru U dari matriks jarak D.

    Rumus: u_ij = 1 / sum_k (d_ij / d_ik)^(2/(m-1))
    Bentuk setara yang dipakai di kode (lebih stabil secara numerik):
        u_ij = d_ij^(-2/(m-1)) / sum_k d_ik^(-2/(m-1))

    Kasus khusus d_ij = 0 (data tepat berada di pusat cluster j): rumus di
    atas akan membagi dengan nol. Secara teori data itu milik penuh cluster
    tersebut, sehingga u_ij = 1 dan keanggotaan pada cluster lain = 0.
    Jika ada beberapa pusat berjarak nol (pusat kembar), nilai 1 dibagi rata.
    Mengembalikan U berbentuk (n, c), setiap baris berjumlah 1.
    """
    D = np.asarray(D, dtype=np.float64)
    pangkat = 2.0 / (m - 1.0)
    # Tandai pasangan (i, j) yang jaraknya nol.
    nol = D == 0
    baris_nol = nol.any(axis=1)

    U = np.empty_like(D)
    # Baris normal (tidak ada jarak nol).
    if np.any(~baris_nol):
        Dn = D[~baris_nol]
        # Skala per baris dengan jarak terkecil agar tidak overflow/underflow;
        # rasio d_ij/d_ik tidak berubah sehingga hasil tetap sama.
        Dn = Dn / Dn.min(axis=1, keepdims=True)
        w = Dn ** (-pangkat)                      # d_ij^(-2/(m-1))
        U[~baris_nol] = w / w.sum(axis=1, keepdims=True)
    # Baris dengan jarak nol: keanggotaan penuh ke pusat yang berjarak nol.
    if np.any(baris_nol):
        Z = nol[baris_nol].astype(np.float64)
        U[baris_nol] = Z / Z.sum(axis=1, keepdims=True)
    return U


def objective(U: np.ndarray, D: np.ndarray, m: float = 2.0) -> float:
    """Menghitung fungsi objektif J_m = sum_i sum_j u_ij^m * d_ij^2.

    J_m mengukur total jarak berbobot data ke pusat cluster. FCM berusaha
    meminimalkan nilai ini; selama iterasi nilainya tidak boleh naik.
    """
    return float(np.sum((U ** m) * (D ** 2)))


def fpc(U: np.ndarray) -> float:
    """Menghitung Fuzzy Partition Coefficient: FPC = (1/n) sum_i sum_j u_ij^2.

    Nilainya berada di [1/c, 1]. Mendekati 1 berarti pembagian cluster tegas
    (setiap data jelas milik satu cluster); mendekati 1/c berarti sangat kabur.
    """
    U = np.asarray(U, dtype=np.float64)
    return float(np.sum(U ** 2) / U.shape[0])


def dominant_cluster(U: np.ndarray) -> np.ndarray:
    """Mengembalikan indeks cluster dominan (keanggotaan terbesar) tiap data."""
    return np.argmax(U, axis=1)


# ---------------------------------------------------------------------------
# Fungsi utama (kontrak PRD)
# ---------------------------------------------------------------------------
def fcm(
    X: np.ndarray,
    c: int,
    m: float = 2.0,
    error: float = 1e-5,
    maxiter: int = 300,
    seed: int = 42,
    init: np.ndarray | None = None,
) -> dict:
    """Menjalankan algoritma Fuzzy C-Means.

    Parameter
    ---------
    X : array (n, f)   data yang sudah dinormalisasi.
    c : int            jumlah cluster (2 <= c <= n).
    m : float          fuzzifier (> 1), default 2.0.
    error : float      batas konvergensi epsilon untuk max|U_baru - U_lama|.
    maxiter : int      jumlah iterasi maksimum.
    seed : int         seed untuk U awal acak (hasil dapat diulang).
    init : array (n, c), opsional
        U awal yang ditentukan sendiri (dipakai pada contoh hitung manual dan
        saat membandingkan dengan skfuzzy). Jika None, U awal dibuat acak.

    Mengembalikan dict:
        'V' (c, f), 'U' (n, c), 'J_history' list[float], 'n_iter' int, 'fpc' float

    Urutan per iterasi t:
        1. V(t)  = update_centers(X, U(t-1))
        2. D(t)  = jarak setiap data ke V(t)
        3. J(t)  = sum u_ij(t-1)^m * d_ij(t)^2   (dicatat di J_history)
        4. U(t)  = update_membership(D(t))
        5. jika max|U(t) - U(t-1)| < error -> berhenti.
    Urutan ini sama dengan skfuzzy.cmeans sehingga J_history dapat dibandingkan
    langsung dan dijamin tidak naik.
    """
    # Validasi semua masukan dengan pesan berbahasa Indonesia.
    X = _validasi_X(X)
    n = X.shape[0]
    c = _validasi_c(c, n)
    m = _validasi_m(m)
    if isinstance(error, bool) or not isinstance(error, numbers.Real) or not error > 0:
        raise ValueError(f"error (epsilon) harus angka positif, diberikan {error!r}.")
    if isinstance(maxiter, bool) or not isinstance(maxiter, numbers.Integral) or maxiter < 1:
        raise ValueError(f"maxiter harus bilangan bulat >= 1, diberikan {maxiter!r}.")

    # Langkah 1: U awal (acak dengan seed, atau diberikan lewat init).
    if init is None:
        U = init_membership(n, c, seed)
    else:
        U = np.asarray(init, dtype=np.float64)
        if U.shape != (n, c):
            raise ValueError(f"init harus berbentuk (n, c) = ({n}, {c}), diberikan {U.shape}.")
        if not np.all(np.isfinite(U)) or np.any(U < 0):
            raise ValueError("init harus berisi angka berhingga yang tidak negatif.")
        if np.any(U.sum(axis=1) == 0):
            raise ValueError("Setiap baris init harus memiliki jumlah lebih dari 0.")
        # Pastikan setiap baris berjumlah 1.
        U = U / U.sum(axis=1, keepdims=True)

    J_history: list[float] = []
    n_iter = 0
    for n_iter in range(1, int(maxiter) + 1):
        U_lama = U
        # Langkah 2: pusat cluster dari keanggotaan saat ini.
        V = update_centers(X, U_lama, m)
        # Langkah 3: jarak setiap data ke setiap pusat.
        D = distances(X, V)
        # Catat fungsi objektif (keanggotaan lama, pusat baru).
        J_history.append(objective(U_lama, D, m))
        # Langkah 4: keanggotaan baru dari jarak.
        U = update_membership(D, m)
        # Langkah 5: cek konvergensi dengan perubahan terbesar pada U.
        if np.max(np.abs(U - U_lama)) < error:
            break

    # V yang dikembalikan adalah pusat dari iterasi terakhir, dan U adalah
    # keanggotaan yang dihitung dari V tersebut (sama seperti skfuzzy).
    # Akibatnya membership(X, V, m) menghasilkan U yang sama persis.
    return {
        "V": V,
        "U": U,
        "J_history": J_history,
        "n_iter": int(n_iter),
        "fpc": fpc(U),
    }


def membership(x: np.ndarray, V: np.ndarray, m: float = 2.0) -> np.ndarray:
    """Menghitung derajat keanggotaan data (baru) terhadap pusat V yang sudah ada.

    Dipakai FuzzyFlix untuk film baru atau film input pengguna: pusat V hasil
    pelatihan tidak diubah, hanya rumus u_ij yang diterapkan.

    x : array (f,) untuk satu data -> hasil (c,)
        atau array (k, f) untuk banyak data -> hasil (k, c)
    """
    V = _validasi_X(V)
    m = _validasi_m(m)
    x_arr = np.asarray(x, dtype=np.float64)
    satu_data = x_arr.ndim == 1
    X2 = _validasi_X(x_arr[None, :] if satu_data else x_arr)
    if X2.shape[1] != V.shape[1]:
        raise ValueError(
            f"Jumlah fitur x ({X2.shape[1]}) tidak sama dengan jumlah fitur V ({V.shape[1]})."
        )
    U = update_membership(distances(X2, V), m)
    return U[0] if satu_data else U


def select_c(X, c_range=range(3, 9), **kw) -> pd.DataFrame:
    """Mencoba beberapa nilai c dan menghitung FPC serta silhouette.

    - FPC dihitung dari U (semakin besar semakin tegas).
    - Silhouette dihitung dengan sklearn pada label cluster dominan
      (argmax U), memakai jarak Euclidean pada X.
    kw diteruskan ke fcm() (misalnya m, error, maxiter, seed).
    Mengembalikan DataFrame dengan kolom: c, fpc, silhouette.
    """
    from sklearn.metrics import silhouette_score

    X = _validasi_X(X)
    baris = []
    for c in c_range:
        hasil = fcm(X, c, **kw)
        label = dominant_cluster(hasil["U"])
        # Silhouette hanya terdefinisi jika 2 <= jumlah label unik <= n-1.
        k = len(np.unique(label))
        sil = float(silhouette_score(X, label)) if 2 <= k <= X.shape[0] - 1 else float("nan")
        baris.append({"c": int(c), "fpc": hasil["fpc"], "silhouette": sil})
    return pd.DataFrame(baris, columns=["c", "fpc", "silhouette"])
