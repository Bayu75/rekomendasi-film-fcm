"""Contoh perhitungan sederhana FCM dan KNN fuzzy (dihitung langsung, bukan teks tetap).

Anggota 4: GUI & Visualisasi. Modul ini murni NumPy/pandas (tanpa ``st.*``).
Tujuannya memenuhi ketentuan F.7 tugas: menunjukkan pemahaman algoritma lewat
contoh kecil (5 film × 2 fitur × 2 cluster) yang angkanya bisa dicek dengan
kalkulator. Rumus sama persis dengan PRD bagian 5.2 dan 5.3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAMA_FILM = ["A", "B", "C", "D", "E"]
NAMA_FITUR = ["Rating (0–1)", "Popularitas (0–1)"]

# Dua kelompok jelas (A, B tinggi; D, E rendah) dan satu film di antaranya (C).
X_CONTOH = np.array([[0.90, 0.80],
                     [0.80, 0.90],
                     [0.50, 0.50],
                     [0.20, 0.30],
                     [0.10, 0.20]])
# Keanggotaan awal U^(0): tiap baris berjumlah 1.
U0_CONTOH = np.array([[0.90, 0.10],
                      [0.80, 0.20],
                      [0.60, 0.40],
                      [0.20, 0.80],
                      [0.10, 0.90]])


def tabel_data_contoh() -> pd.DataFrame:
    """Data contoh bawaan sebagai tabel yang bisa diedit pengguna."""
    d = pd.DataFrame(X_CONTOH, columns=NAMA_FITUR)
    d.insert(0, "Film", NAMA_FILM)
    return d


def jarak(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Jarak Euclidean tiap film ke tiap pusat, shape (n, c); minimal 1e-12 agar tidak membagi nol."""
    d = np.sqrt(((X[:, None, :] - V[None, :, :]) ** 2).sum(axis=2))
    return np.fmax(d, 1e-12)


def normalisasi_baris(U: np.ndarray) -> np.ndarray:
    """Bagi tiap baris dengan jumlahnya sehingga Σ_j u_ij = 1."""
    U = np.asarray(U, dtype=float)
    return U / U.sum(axis=1, keepdims=True)


def langkah_fcm(X: np.ndarray, U0: np.ndarray, m: float = 2.0, n_iter: int = 2) -> list[dict]:
    """Jalankan ``n_iter`` iterasi FCM dan simpan semua hasil antara.

    Tiap iterasi (PRD 5.2):
      1. ``V = Σ u^m x / Σ u^m``                       (pusat cluster)
      2. ``d_ij = ‖x_i − v_j‖``                         (jarak)
      3. ``u_ij = 1 / Σ_k (d_ij/d_ik)^(2/(m−1))``       (keanggotaan baru)
      4. ``J = Σ u_ij^m d_ij²``,  ``Δ = max|U_baru − U_lama|``

    Return: daftar dict berkunci ``iter, U_in, num, den, V, d, U_out, J, delta``.
    """
    X = np.asarray(X, dtype=float)
    U = normalisasi_baris(U0)
    hasil = []
    for t in range(1, n_iter + 1):
        Um = U ** m
        num = Um.T @ X                       # (c, f)
        den = Um.sum(axis=0)                 # (c,)
        V = num / den[:, None]
        d = jarak(X, V)
        p = 2.0 / (m - 1.0)
        U_baru = 1.0 / ((d[:, :, None] / d[:, None, :]) ** p).sum(axis=2)
        J = float(((U_baru ** m) * d ** 2).sum())
        hasil.append({"iter": t, "U_in": U, "num": num, "den": den, "V": V, "d": d,
                      "U_out": U_baru, "J": J, "delta": float(np.abs(U_baru - U).max())})
        U = U_baru
    return hasil


def teks_pusat(X: np.ndarray, U: np.ndarray, m: float, j: int, f: int) -> str:
    """LaTeX perhitungan ``v_{j,f}`` dengan semua angka disubstitusi (untuk satu cluster dan satu fitur)."""
    w = U[:, j] ** m
    pembilang = " + ".join(f"({U[i, j]:.3f}^{{{m:g}}} \\times {X[i, f]:.2f})" for i in range(len(X)))
    penyebut = " + ".join(f"{U[i, j]:.3f}^{{{m:g}}}" for i in range(len(X)))
    nilai = float((w * X[:, f]).sum() / w.sum())
    return (rf"v_{{{j + 1},{f + 1}}} = \frac{{{pembilang}}}{{{penyebut}}} "
            rf"= \frac{{{(w * X[:, f]).sum():.4f}}}{{{w.sum():.4f}}} = {nilai:.4f}")


def teks_keanggotaan(d: np.ndarray, m: float, i: int, j: int, nama_film: str = "A") -> str:
    """LaTeX perhitungan ``u_{ij}`` dengan angka jarak disubstitusi (c = 2 atau lebih)."""
    p = 2.0 / (m - 1.0)
    c = d.shape[1]
    suku = " + ".join(f"\\left(\\frac{{{d[i, j]:.4f}}}{{{d[i, k]:.4f}}}\\right)^{{{p:g}}}" for k in range(c))
    nilai = 1.0 / sum((d[i, j] / d[i, k]) ** p for k in range(c))
    return rf"u_{{{nama_film},{j + 1}}} = \frac{{1}}{{{suku}}} = {nilai:.4f}"


def jarak_knn_contoh(X: np.ndarray, U: np.ndarray, i: int, k: int, alpha: float = 0.5) -> dict:
    """Jarak KNN fuzzy antara film ``i`` (acuan) dan film ``k`` (PRD 5.3).

    ``d = α·‖x_i − x_k‖ + (1−α)·‖u_i − u_k‖``; ``skor = 1/(1+d)``.
    """
    d_fitur = float(np.linalg.norm(X[i] - X[k]))
    d_mem = float(np.linalg.norm(U[i] - U[k]))
    d = alpha * d_fitur + (1 - alpha) * d_mem
    return {"d_fitur": d_fitur, "d_mem": d_mem, "bagian_fitur": alpha * d_fitur,
            "bagian_mem": (1 - alpha) * d_mem, "d": d, "skor": 1.0 / (1.0 + d)}


def peringkat_knn_contoh(X: np.ndarray, U: np.ndarray, q: int, alpha: float) -> pd.DataFrame:
    """Urutan semua film lain terhadap film acuan ``q`` untuk nilai α tertentu."""
    baris = []
    for k in range(len(X)):
        if k == q:
            continue
        r = jarak_knn_contoh(X, U, q, k, alpha)
        baris.append({"idx": k, **r})
    return pd.DataFrame(baris).sort_values("d").reset_index(drop=True)
