"""Teks interpretasi otomatis, pesan error GUI, dan utilitas tampilan tabel film.

Anggota 4: GUI & Visualisasi. Modul ini murni (tanpa ``st.*``) sehingga bisa
dites dan dipakai untuk laporan. Istilah mengikuti design-system.md bagian 8:
*derajat keanggotaan*, *cluster*, *pusat cluster*, *film acuan*,
*skor kemiripan*, *KNN biasa*.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

KOLOM_WAJIB = ["id", "title", "vote_average", "vote_count", "popularity",
               "runtime", "release_date", "genres"]

# Rentang wajar tiap variabel (dipakai form input manual dan pesan error).
RENTANG = {
    "vote_average": (0.0, 10.0),
    "vote_count": (50, 15000),
    "popularity": (0.0, 550.0),
    "runtime": (1, 400),
    "year": (1874, 2026),
}

PENJELASAN_VARIABEL = pd.DataFrame([
    ("Rating", "vote_average", "Rata-rata nilai dari penonton (0–10).", "Min-max 0–1"),
    ("Jumlah vote", "vote_count", "Banyaknya penonton yang memberi nilai; menandakan seberapa dikenal sebuah film.", "log1p lalu min-max"),
    ("Popularitas", "popularity", "Skor popularitas TMDB (makin tinggi makin sering dicari/ditonton).", "log1p lalu min-max"),
    ("Durasi", "runtime", "Lama film dalam menit.", "Min-max"),
    ("Tahun rilis", "release_date", "Tahun film pertama kali dirilis, diambil dari tanggal rilis.", "Ambil tahun lalu min-max"),
    ("Genre", "genres", "Satu atau beberapa genre film (20 genre di dataset).", "One-hot 20 kolom lalu PCA (±80% variansi)"),
], columns=["Variabel", "Kolom CSV", "Penjelasan", "Perlakuan"])


# ---------------------------------------------------------------------------
# Genre & tabel film
# ---------------------------------------------------------------------------

def parse_genres(nilai) -> list[str]:
    """Ubah nilai kolom genre apa pun menjadi daftar nama genre.

    Format yang diterima: list/tuple/ndarray, string JSON dataset Kaggle
    (``[{'id': 28, 'name': 'Action'}]``), atau string dipisah ``|`` / ``,``.
    """
    if nilai is None:
        return []
    if isinstance(nilai, float) and np.isnan(nilai):
        return []
    if isinstance(nilai, (list, tuple, np.ndarray, pd.Series)):
        hasil = []
        for e in list(nilai):
            hasil.extend(parse_genres(e) if not isinstance(e, str) else [e.strip()])
        return [g for g in hasil if g]
    if isinstance(nilai, dict):
        return [str(nilai["name"])] if "name" in nilai else []
    teks = str(nilai).strip()
    if not teks or teks.lower() in {"nan", "none", "[]"}:
        return []
    if teks[0] in "[{":
        for loader in (json.loads, ast.literal_eval):
            try:
                return parse_genres(loader(teks))
            except Exception:  # noqa: BLE001 - coba parser berikutnya
                continue
    return [g.strip() for g in re.split(r"[|,;]", teks) if g.strip()]


def film_year(df: pd.DataFrame) -> pd.Series:
    """Tahun rilis tiap film: kolom ``year``/``tahun``, atau dari ``release_date``."""
    for k in ("year", "tahun", "release_year"):
        if k in df.columns:
            return pd.to_numeric(df[k], errors="coerce")
    if "release_date" in df.columns:
        return pd.to_datetime(df["release_date"], errors="coerce").dt.year.astype(float)
    return pd.Series(np.nan, index=df.index)


def siapkan_tabel_film(df_film: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom bantu tampilan: ``_year``, ``_genres`` (list), ``_genres_text``.

    Tidak mengubah ``df_film`` asli. Dipakai oleh semua halaman supaya GUI tidak
    bergantung pada format genre/tahun yang dipilih Anggota 1.
    """
    d = df_film.reset_index(drop=True).copy()
    d["_year"] = film_year(d)
    d["_genres"] = d["genres"].apply(parse_genres) if "genres" in d.columns else [[] for _ in range(len(d))]
    d["_genres_text"] = d["_genres"].apply(lambda g: ", ".join(g[:3]) + ("…" if len(g) > 3 else ""))
    return d


def fmt_int(n: int | float) -> str:
    """Angka bulat dengan pemisah titik ribuan: 9089 -> ``9.089``."""
    return f"{int(n):,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Penamaan & profil cluster
# ---------------------------------------------------------------------------

def name_clusters(U: np.ndarray, film: pd.DataFrame, m: float = 2.0) -> list[str]:
    """Beri nama tiap cluster dari cirinya (PRD 5.2): genre dominan, rating, tahun.

    Rating dan tahun adalah rata-rata berbobot ``u_ij ** m`` sehingga film yang
    keanggotaannya kuat lebih menentukan nama. Genre dipilih dari genre yang paling
    *khas* (bobot di cluster × lift terhadap seluruh film). ``film`` = hasil
    ``siapkan_tabel_film``. Format nama: ``C2 · Action (6.4, 1998)``.
    """
    U = np.asarray(U, dtype=float)
    c = U.shape[1]
    rating = pd.to_numeric(film.get("vote_average"), errors="coerce").to_numpy(dtype=float)
    tahun = film["_year"].to_numpy(dtype=float)
    genres = film["_genres"].tolist()
    # genre khas = bobot genre di cluster dikalikan lift terhadap seluruh film,
    # supaya genre umum (mis. Drama) tidak menenggelamkan genre pembeda.
    global_cnt: Counter = Counter(g for gs in genres for g in gs)
    n_film = max(len(genres), 1)
    nama = []
    for j in range(c):
        w = U[:, j] ** m
        cnt: Counter = Counter()
        for wi, gs in zip(w, genres):
            for g in gs:
                cnt[g] += wi
        total_w = max(w.sum(), 1e-12)
        skor = {g: (v / total_w) * ((v / total_w) / (global_cnt[g] / n_film)) for g, v in cnt.items()}
        top = max(skor, key=skor.get) if skor else "Campuran"
        r = _wmean(rating, w)
        t = _wmean(tahun, w)
        bagian = []
        if np.isfinite(r):
            bagian.append(f"{r:.1f}")
        if np.isfinite(t):
            bagian.append(f"{t:.0f}")
        nama.append(f"C{j + 1} · {top} ({', '.join(bagian)})" if bagian else f"C{j + 1} · {top}")
    return nama


def _wmean(v: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(v)
    if not ok.any() or w[ok].sum() <= 0:
        return float("nan")
    return float(np.average(v[ok], weights=w[ok]))


def fitur_baca(film: pd.DataFrame) -> pd.DataFrame:
    """Ciri film dalam satuan asli (untuk radar): Rating, Jumlah vote, Popularitas, Durasi, Tahun rilis.

    Kolom yang tidak ada di ``df_film`` dilewati, sehingga GUI tidak bergantung
    pada urutan kolom ``X`` yang dibuat Anggota 1.
    """
    peta = {"Rating": "vote_average", "Jumlah vote": "vote_count",
            "Popularitas": "popularity", "Durasi": "runtime", "Tahun rilis": "_year"}
    return pd.DataFrame({k: pd.to_numeric(film[v], errors="coerce")
                         for k, v in peta.items() if v in film.columns})


def cluster_profile(U: np.ndarray, film: pd.DataFrame, m: float = 2.0) -> pd.DataFrame:
    """Pusat cluster dalam satuan asli (rata-rata berbobot ``u^m``) untuk radar (F-19)."""
    U = np.asarray(U, dtype=float)
    kolom = {k: v.to_numpy(dtype=float) for k, v in fitur_baca(film).items()}
    baris = []
    for j in range(U.shape[1]):
        w = U[:, j] ** m
        baris.append({k: _wmean(v, w) for k, v in kolom.items()})
    return pd.DataFrame(baris)


# ---------------------------------------------------------------------------
# Interpretasi otomatis (design-system 8, F-18)
# ---------------------------------------------------------------------------

def genre_dominan(genres_list: Iterable[Sequence[str]], top: int = 2) -> str:
    """Genre paling sering muncul pada sekumpulan film, mis. ``Action dan Adventure``."""
    cnt = Counter(g for gs in genres_list for g in gs)
    if not cnt:
        return "bergenre campuran"
    g = [k for k, _ in cnt.most_common(top)]
    return " dan ".join(g)


def interpretasi_rekomendasi(judul: str, u_q: np.ndarray, nama_cluster: Sequence[str],
                             rec_genres: Iterable[Sequence[str]], rating_rata: float,
                             alpha: float | None = None) -> str:
    """Kalimat interpretasi otomatis halaman Hasil (template design-system bagian 8).

    Template:
        "{judul} paling kuat berada di cluster {nama_cluster} ({u_max:.2f}) dan juga
        memiliki keanggotaan {u_2:.2f} di cluster {nama_cluster_2}. Karena itu,
        rekomendasi didominasi film {genre_dominan} dengan rating rata-rata {rating_rata:.1f}."

    Jika film hanya punya satu cluster yang bermakna (u_2 < 0,10), kalimat kedua
    disesuaikan. Jika ``alpha`` = 1, ditambahkan catatan bahwa hasil sama dengan KNN biasa.
    """
    u = np.asarray(u_q, dtype=float).ravel()
    urut = np.argsort(-u)
    j1 = int(urut[0])
    gd = genre_dominan(rec_genres)
    rr = f"{rating_rata:.1f}" if np.isfinite(rating_rata) else "-"
    if len(u) > 1:
        j2 = int(urut[1])
        if u[j2] >= 0.10:
            kalimat = (f"**{judul}** paling kuat berada di cluster **{nama_cluster[j1]}** ({u[j1]:.2f}) "
                       f"dan juga memiliki keanggotaan {u[j2]:.2f} di cluster {nama_cluster[j2]}. "
                       f"Karena itu, rekomendasi didominasi film {gd} dengan rating rata-rata {rr}.")
        else:
            kalimat = (f"**{judul}** hampir sepenuhnya berada di cluster **{nama_cluster[j1]}** ({u[j1]:.2f}); "
                       f"keanggotaan tertinggi berikutnya hanya {u[j2]:.2f} di cluster {nama_cluster[j2]}. "
                       f"Karena itu, rekomendasi didominasi film {gd} dengan rating rata-rata {rr}.")
    else:
        kalimat = (f"**{judul}** berada di cluster **{nama_cluster[j1]}** ({u[j1]:.2f}). "
                   f"Rekomendasi didominasi film {gd} dengan rating rata-rata {rr}.")
    if alpha is not None and float(alpha) >= 1.0:
        kalimat += " Karena α = 1, hasil ini sama dengan KNN biasa tanpa pengaruh derajat keanggotaan."
    return kalimat


def interpretasi_evaluasi(p_fuzzy: float, p_base: float, k: int, fpc: float | None,
                          silhouette: float | None) -> list[str]:
    """Poin-poin interpretasi hasil pengujian (F-22), berdasarkan angka yang tampil."""
    poin = []
    if p_fuzzy is not None and p_base is not None:
        selisih = p_fuzzy - p_base
        if selisih > 0.005:
            poin.append(f"Pada K = {k}, KNN fuzzy mencapai Precision {p_fuzzy:.3f} dibanding "
                        f"{p_base:.3f} untuk KNN biasa (selisih +{selisih:.3f}). Derajat keanggotaan "
                        "membantu memilih tetangga yang genre-nya lebih sesuai.")
        elif selisih < -0.005:
            poin.append(f"Pada K = {k}, KNN fuzzy ({p_fuzzy:.3f}) justru lebih rendah dari KNN biasa "
                        f"({p_base:.3f}). Coba ubah α, jumlah cluster, atau fuzzifier m, dan laporkan "
                        "hasil ini apa adanya sebagai keterbatasan.")
        else:
            poin.append(f"Pada K = {k}, KNN fuzzy ({p_fuzzy:.3f}) dan KNN biasa ({p_base:.3f}) "
                        "hampir sama; kontribusi FCM belum terlihat pada ukuran ini.")
    if fpc is not None:
        if fpc >= 0.6:
            poin.append(f"FPC = {fpc:.3f}: partisi cukup tegas (mendekati 1 = tegas).")
        else:
            poin.append(f"FPC = {fpc:.3f}: partisi masih lunak, banyak film punya keanggotaan "
                        "yang terbagi ke beberapa cluster. Ini wajar untuk film berciri campuran.")
    if silhouette is not None:
        if silhouette > 0.25:
            poin.append(f"Silhouette = {silhouette:.3f} (> 0,25): cluster cukup rapi.")
        else:
            poin.append(f"Silhouette = {silhouette:.3f} (≤ 0,25): cluster saling berdekatan karena data padat. "
                        "Laporkan sebagai keterbatasan.")
    return poin


# ---------------------------------------------------------------------------
# Validasi dan pesan error (format: apa yang salah + cara memperbaiki)
# ---------------------------------------------------------------------------

def validasi_csv_cadangan(df: pd.DataFrame) -> list[str]:
    """Validasi CSV unggahan bila ``preprocessing.validate_csv`` belum tersedia.

    Mengembalikan daftar pesan error (kosong = valid), sama seperti kontrak PRD.
    """
    pesan = []
    hilang = [k for k in KOLOM_WAJIB if k not in df.columns]
    if hilang:
        pesan.append("Kolom " + ", ".join(f"`{k}`" for k in hilang) + " tidak ditemukan. "
                     "Pastikan CSV memiliki kolom: " + ", ".join(KOLOM_WAJIB) + ".")
        return pesan
    if len(df) == 0:
        return ["File tidak berisi baris data. Pastikan CSV memuat minimal satu film."]
    for k in ("vote_average", "vote_count", "popularity", "runtime"):
        angka = pd.to_numeric(df[k], errors="coerce")
        rusak = int(angka.isna().sum() - df[k].isna().sum())
        if rusak > 0:
            pesan.append(f"Kolom `{k}` berisi {rusak} nilai yang bukan angka. "
                         "Perbaiki nilai tersebut atau hapus barisnya, lalu unggah ulang.")
        if angka.notna().sum() == 0:
            pesan.append(f"Kolom `{k}` tidak memiliki satu pun nilai angka yang valid.")
    tgl = pd.to_datetime(df["release_date"], errors="coerce")
    if tgl.notna().sum() == 0:
        pesan.append("Kolom `release_date` tidak memiliki tanggal valid. Gunakan format YYYY-MM-DD, "
                     "misalnya 1999-03-31.")
    return pesan


def validasi_film_manual(judul: str, rating: float, vote: float, popularitas: float,
                         durasi: float, tahun: int, genres: Sequence[str]) -> list[str]:
    """Validasi form input manual film baru (F-05, F-07). Kosong = valid."""
    pesan = []
    if not judul or not judul.strip():
        pesan.append("Judul film belum diisi. Ketik judul film baru, misalnya `Film Saya`.")
    if not genres:
        pesan.append("Genre belum dipilih. Pilih minimal satu genre pada kolom Genre.")
    cek = {"Rating": (rating, "vote_average", "0–10"), "Jumlah vote": (vote, "vote_count", f"{RENTANG['vote_count'][0]}–{RENTANG['vote_count'][1]}"),
           "Popularitas": (popularitas, "popularity", f"0–{RENTANG['popularity'][1]:.0f}"),
           "Durasi": (durasi, "runtime", "1–400 menit"), "Tahun rilis": (tahun, "year", f"{RENTANG['year'][0]}–{RENTANG['year'][1]}")}
    for label, (val, kunci, teks) in cek.items():
        lo, hi = RENTANG[kunci]
        if val is None or not np.isfinite(val) or not (lo <= val <= hi):
            pesan.append(f"{label} = {val} di luar rentang yang valid. Isi dengan angka {teks}.")
    return pesan


def validasi_parameter(c: int, m: float, error: float, maxiter: int) -> list[str]:
    """Validasi parameter FCM (rentang PRD 5.2). Kosong = valid."""
    pesan = []
    if not 2 <= c <= 10:
        pesan.append(f"Jumlah cluster c = {c} di luar rentang. Pilih c antara 2 dan 10.")
    if not 1.1 <= m <= 3.0:
        pesan.append(f"Fuzzifier m = {m} di luar rentang. Pilih m antara 1,1 dan 3,0.")
    if not 1e-6 <= error <= 1e-3:
        pesan.append(f"Toleransi ε = {error} di luar rentang. Pilih antara 1e-6 dan 1e-3.")
    if not 50 <= maxiter <= 1000:
        pesan.append(f"Iterasi maksimum = {maxiter} di luar rentang. Pilih antara 50 dan 1000.")
    return pesan


def pesan_error_ramah(exc: BaseException, konteks: str = "") -> str:
    """Terjemahkan exception menjadi pesan ramah (apa yang salah + cara memperbaiki).

    Tidak menampilkan traceback; detail teknis ditaruh pemanggil di expander.
    """
    awal = f"{konteks} " if konteks else ""
    if isinstance(exc, FileNotFoundError):
        return (f"{awal}File tidak ditemukan. Pastikan `movies_metadata.csv` ada di folder `data/raw/` "
                "atau jalankan `scripts/train.py` agar folder `models/` terisi.")
    if isinstance(exc, NotImplementedError):
        return (f"{awal}Fungsi yang dibutuhkan belum diimplementasikan oleh anggota tim terkait. "
                "Aplikasi memakai data demo selama fungsi itu belum tersedia.")
    if isinstance(exc, pd.errors.EmptyDataError):
        return f"{awal}File CSV kosong. Pilih file yang berisi data film, lalu unggah ulang."
    if isinstance(exc, pd.errors.ParserError):
        return (f"{awal}File tidak bisa dibaca sebagai CSV. Pastikan file berformat CSV dengan pemisah "
                "koma dan tanda kutip yang tertutup, lalu unggah ulang.")
    if isinstance(exc, UnicodeDecodeError):
        return f"{awal}Encoding file tidak dikenali. Simpan ulang CSV dengan encoding UTF-8, lalu unggah ulang."
    if isinstance(exc, MemoryError):
        return f"{awal}Memori tidak cukup. Gunakan dataset yang lebih kecil atau kurangi jumlah cluster."
    if isinstance(exc, (ValueError, KeyError, IndexError)):
        return (f"{awal}Data atau parameter tidak sesuai ({type(exc).__name__}: {exc}). "
                "Periksa kembali input Anda, lalu coba lagi.")
    return f"{awal}Terjadi kesalahan tak terduga ({type(exc).__name__}). Coba ulangi; jika berlanjut, hubungi tim pengembang."


# ---------------------------------------------------------------------------
# Integrasi FCM → KNN dan prediksi rating (dihitung GUI dari X, U, hasil recommend)
# ---------------------------------------------------------------------------

def rincian_jarak(X: np.ndarray, U: np.ndarray, x_q: np.ndarray, u_q: np.ndarray,
                  idx: Sequence[int], alpha: float) -> pd.DataFrame:
    """Uraikan jarak KNN fuzzy tiap film rekomendasi menjadi dua bagian (PRD 5.3).

    ``d = α·‖x_q − x_i‖ + (1−α)·‖u_q − u_i‖`` dan ``skor = 1/(1+d)``.
    Kolom: ``idx, d_fitur, d_mem, bagian_fitur, bagian_mem, d, skor``. Dipakai halaman
    Hasil untuk memperlihatkan secara eksplisit bagaimana vektor keanggotaan FCM masuk
    ke dalam jarak KNN, dan untuk memeriksa bahwa hasil modul KNN konsisten.
    """
    idx = np.asarray(list(idx), dtype=int)
    d_fitur = np.linalg.norm(np.asarray(X)[idx] - np.asarray(x_q), axis=1)
    d_mem = np.linalg.norm(np.asarray(U)[idx] - np.asarray(u_q), axis=1)
    bf, bm = alpha * d_fitur, (1 - alpha) * d_mem
    d = bf + bm
    return pd.DataFrame({"idx": idx, "d_fitur": d_fitur, "d_mem": d_mem, "bagian_fitur": bf,
                         "bagian_mem": bm, "d": d, "skor": 1.0 / (1.0 + d)})


def prediksi_rating(rating_tetangga: np.ndarray, skor: np.ndarray) -> float:
    """Prediksi rating film acuan = rata-rata rating tetangga berbobot skor kemiripan.

    Ini adaptasi grafik "aktual vs prediksi sistem" untuk sistem rekomendasi:
    jika tetangga terdekat benar-benar mirip, rating mereka mendekati rating asli film acuan.
    """
    r = np.asarray(rating_tetangga, dtype=float)
    w = np.asarray(skor, dtype=float)
    ok = np.isfinite(r)
    if not ok.any() or w[ok].sum() <= 0:
        return float("nan")
    return float(np.average(r[ok], weights=w[ok]))


def galat_prediksi(aktual: np.ndarray, prediksi: np.ndarray) -> tuple[float, float]:
    """(RMSE, MAE) antara rating aktual dan prediksi; baris NaN diabaikan."""
    a, p = np.asarray(aktual, dtype=float), np.asarray(prediksi, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    if not ok.any():
        return float("nan"), float("nan")
    e = a[ok] - p[ok]
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))
