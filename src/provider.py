"""Penyedia data untuk GUI: satu-satunya tempat GUI bertemu modul anggota lain.

Anggota 4: GUI & Visualisasi.

Semua halaman memanggil fungsi di modul ini, bukan ``src.fcm`` dst. secara
langsung. Setiap pemanggilan mengikuti kontrak antarmodul PRD bagian 8:

    modul asli lebih dulu  ->  jika belum ada / masih ``NotImplementedError``
                               ->  pengganti di ``src/demo_data.py`` (diberi label)

Jadi saat modul asli selesai, GUI otomatis memakainya tanpa perubahan kode
(lihat docs/GUI-notes.md). Fungsi yang jatuh ke demo dicatat di
``st.session_state.bagian_demo`` dan ditampilkan sebagai banner ``st.info``.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from src import demo_data, interpret

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_RAW = ROOT / "data" / "raw"
CSV_BAWAAN = DATA_RAW / "movies_metadata.csv"
ARTEFAK_FILE = ("X.npy", "U.npy", "V.npy", "artefak.joblib", "film.parquet")

DEFAULT_PARAM = {"c": 5, "m": 2.0, "error": 1e-5, "maxiter": 300}


class BackendBelumSiap(RuntimeError):
    """Fungsi modul asli belum diimplementasikan dan tidak ada pengganti demo."""


# ---------------------------------------------------------------------------
# Pemanggilan modul asli dengan cadangan demo
# ---------------------------------------------------------------------------

# Fungsi asli yang memakai kontrak PRD bagian 8 (dibuat Anggota 3).
NAMA_PRD = {("knn_fuzzy", "recommend"): "recommend_prd",
            ("evaluation", "precision_at_k"): "precision_at_k_prd"}


def _fn_asli(modul: str, nama: str) -> Callable | None:
    """Ambil fungsi ``src.<modul>.<nama>`` jika modulnya bisa diimpor."""
    try:
        mod = importlib.import_module(f"src.{modul}")
    except Exception:  # noqa: BLE001 - modul belum ada / error impor
        return None
    return getattr(mod, NAMA_PRD.get((modul, nama), nama), None)


def _panggil(modul: str, nama: str, *args, **kw) -> tuple[Any, bool]:
    """Panggil fungsi asli; jika belum diimplementasikan pakai ``demo_data``.

    Return ``(hasil, dari_demo)``. Exception lain (ValueError, dst.) tidak
    ditangkap di sini; halaman menampilkannya sebagai ``st.error`` ramah.
    """
    asli = _fn_asli(modul, nama)
    if asli is not None:
        try:
            return asli(*args, **kw), False
        except NotImplementedError:
            pass
    return getattr(demo_data, nama)(*args, **kw), True


def _tandai(bagian: str, dari_demo: bool) -> None:
    """Catat bagian yang memakai demo, supaya banner jujur."""
    s = st.session_state.setdefault("bagian_demo", set())
    (s.add if dari_demo else s.discard)(bagian)


# ---------------------------------------------------------------------------
# Status sesi
# ---------------------------------------------------------------------------

def _token(*arr) -> str:
    h = hashlib.md5()
    for a in arr:
        h.update(np.ascontiguousarray(a).tobytes() if isinstance(a, np.ndarray) else str(a).encode())
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def _muat_artefak_model() -> dict | None:
    """Data asli tim: X & film dari ``data_loader.load_data()`` (Anggota 1),
    U & V dari ``load_model()`` + ``pca_fcm.joblib`` (Anggota 2).
    Return None (-> GUI memakai data demo) bila salah satu belum ada / tidak cocok."""
    try:
        import json
        import joblib
        from src.data_loader import load_data, load_model
        if not (MODELS_DIR / "pca_fcm.joblib").exists():
            return None
        X, G, movies, _ = load_data()
        U, V, asli = load_model(len(X))
        if not asli:
            return None
        pca = joblib.load(MODELS_DIR / "pca_fcm.joblib")
        prep_path = ROOT / "data" / "processed" / "preprocessing_artifacts.pkl"
        prep = joblib.load(prep_path) if prep_path.exists() else None
        meta_path = MODELS_DIR / "fcm_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except Exception:  # noqa: BLE001
        return None
    return {"X": X, "U": U, "V": V, "df_film": movies, "meta": meta,
            "artefak": {"pca_fcm": pca, "pca_2d": pca, "prep": prep, "G": G}}


@st.cache_data(show_spinner=False)
def _data_demo() -> tuple:
    """Bangun dataset demo sekali per proses (X, df_film, artefak)."""
    raw = demo_data.buat_data_mentah()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "demo.csv"
        raw.to_csv(p, index=False)
        bersih = demo_data.load_and_clean(str(p))
    X, df_film, art = demo_data.build_features(bersih)
    return X, df_film, art, _ekstra_film(bersih)


def _ekstra_film(sumber) -> pd.DataFrame | None:
    """Ambil kolom opsional ``poster_path`` dan ``overview`` (tampilan saja) dari CSV/DataFrame.

    Kolom ini bukan fitur model. Bila tidak ada, GUI memakai poster pengganti dan
    tanpa sinopsis. Hasil: DataFrame ``id, poster_path, overview, _kunci`` atau None.
    """
    try:
        if isinstance(sumber, pd.DataFrame):
            df = sumber
        else:
            df = pd.read_csv(sumber, usecols=lambda c: c in ("id", "poster_path", "overview"), low_memory=False)
        if "id" not in df.columns or not ({"poster_path", "overview"} & set(df.columns)):
            return None
        e = df[[c for c in ("id", "poster_path", "overview") if c in df.columns]].copy()
        e["_kunci"] = e["id"].astype(str)
        return e.drop_duplicates("_kunci")
    except Exception:  # noqa: BLE001 - pelengkap tampilan, tidak boleh menggagalkan pemuatan
        return None


def _pasang_data(X, df_film, artefak, sumber: str, mode: str, ekstra: pd.DataFrame | None = None) -> None:
    """Pasang data baru ke session_state dan buang hasil turunan yang usang."""
    ss = st.session_state
    ss.X, ss.df_film, ss.artefak = X, df_film.reset_index(drop=True), artefak
    ss.film_view = interpret.siapkan_tabel_film(ss.df_film)
    fv = ss.film_view
    for kol in ("poster_path", "overview"):
        if kol in ss.df_film.columns:
            fv[kol] = ss.df_film[kol].to_numpy()
    if ekstra is not None and "id" in fv.columns:
        kunci = fv["id"].astype(str)
        peta = ekstra.set_index("_kunci")
        for kol in ("poster_path", "overview"):
            if kol in peta.columns and kol not in fv.columns:
                fv[kol] = kunci.map(peta[kol]).to_numpy()
    for kol in ("poster_path", "overview"):
        if kol not in fv.columns:
            fv[kol] = None
    ss.data_mode, ss.data_sumber = mode, sumber
    ss.data_ver = _token(X.shape, sumber, len(df_film))
    for k in ("fcm_result", "last_reco", "new_film", "select_c_df", "eval_hasil", "nama_cluster", "hasil_antara"):
        ss[k] = None


def _pasang_model(hasil: dict, param: dict) -> None:
    ss = st.session_state
    ss.fcm_result = hasil
    ss.fcm_param = dict(param)
    ss.nama_cluster = interpret.name_clusters(hasil["U"], ss.film_view, param["m"])
    ss.last_reco = None
    ss.eval_hasil = None
    ss.hasil_antara = None
    _segarkan_film_baru()


def _pasang_asli(arts: dict) -> None:
    """Pasang data + model asli (U, V Anggota 2) ke sesi."""
    _pasang_data(arts["X"], arts["df_film"], arts["artefak"], "data/processed + models/ (U, V asli)", "real")
    U, V, meta = arts["U"], arts["V"], arts["meta"]
    hasil = {"V": V, "U": U, "J_history": meta.get("J_history", []), "n_iter": meta.get("n_iter"),
             "fpc": float((U ** 2).sum() / len(U))}
    _pasang_model(hasil, {**DEFAULT_PARAM, "c": int(V.shape[0]), "m": float(meta.get("m", 2.0))})
    st.session_state.bagian_demo = set()
    try:
        hitung_hasil_antara()          # butuh n_iter dari fcm_meta.json
    except Exception:  # noqa: BLE001
        st.session_state.hasil_antara = None


def init_session() -> None:
    """Siapkan session_state (idempoten). Dipanggil di setiap halaman.

    Urutan: artefak ``models/`` asli -> jika tidak ada, dataset demo dengan
    model FCM demo (c = 5) agar semua halaman langsung bisa dirender.
    """
    ss = st.session_state
    if "data_mode" in ss:
        return
    ss.setdefault("bagian_demo", set())
    arts = _muat_artefak_model()
    if arts is not None:
        _pasang_asli(arts)
        return
    X, df, art, eks = _data_demo()
    _pasang_data(X, df, art, "demo sintetis (1.200 film karangan)", "demo", ekstra=eks)
    try:
        jalankan_clustering(**DEFAULT_PARAM)
        hitung_pilih_c(DEFAULT_PARAM["m"], DEFAULT_PARAM["error"], DEFAULT_PARAM["maxiter"])
    except Exception:  # noqa: BLE001 - jangan gagalkan halaman; pengguna bisa klik tombol
        pass


def data_siap() -> bool:
    return st.session_state.get("df_film") is not None


def model_siap() -> bool:
    return st.session_state.get("fcm_result") is not None


def dari_demo() -> bool:
    """True jika ada bagian data/algoritma yang berasal dari demo."""
    ss = st.session_state
    return ss.get("data_mode") == "demo" or bool(ss.get("bagian_demo"))


def info_demo() -> list[str]:
    """Kalimat penjelasan untuk banner demo (kosong jika semua asli)."""
    ss = st.session_state
    baris = []
    if ss.get("data_mode") == "demo":
        if str(ss.get("data_sumber", "")).startswith("demo sintetis"):
            baris.append("Dataset: **data demo sintetis** (judul film karangan, bukan The Movies Dataset).")
        else:
            baris.append(f"Dataset: **{ss.data_sumber}**, diproses dengan modul pengganti demo "
                         "karena `preprocessing.py` belum selesai.")
    if ss.get("bagian_demo"):
        nama = {"load_and_clean": "pemuatan data", "build_features": "pembentukan fitur",
                "validate_csv": "validasi CSV", "transform_new": "transformasi film baru",
                "fcm": "Fuzzy C-Means", "membership": "keanggotaan film baru", "select_c": "pemilihan c",
                "recommend": "KNN berbobot fuzzy", "precision_at_k": "Precision@K"}
        baris.append("Modul pengganti demo dipakai untuk: " +
                     ", ".join(nama.get(b, b) for b in sorted(ss["bagian_demo"])) + ".")
    return baris


# ---------------------------------------------------------------------------
# Halaman Input Data
# ---------------------------------------------------------------------------

def validasi_csv(df: pd.DataFrame) -> list[str]:
    """Validasi CSV unggahan lewat ``preprocessing.validate_csv`` (atau cadangan)."""
    pesan, demo = _panggil("preprocessing", "validate_csv", df)
    _tandai("validate_csv", demo)
    return list(pesan)


def _cek_jumlah(bersih: pd.DataFrame) -> None:
    """Tolak dataset yang terlalu kecil setelah pembersihan (butuh ≥ 20 film untuk clustering)."""
    if len(bersih) < 20:
        raise ValueError(f"Hanya {len(bersih)} film yang lolos pembersihan (vote_count ≥ 50, runtime > 0, tanggal rilis dan "
                         "genre terisi). Gunakan dataset dengan minimal 20 film yang memenuhi syarat itu.")


def _muat_dari_path(path: str, sumber: str) -> str:
    """Jalankan load_and_clean + build_features pada path, pasang ke sesi."""
    asli_lc, asli_bf = _fn_asli("preprocessing", "load_and_clean"), _fn_asli("preprocessing", "build_features")
    try:
        if asli_lc is None or asli_bf is None:
            raise NotImplementedError
        bersih = asli_lc(path)
        _cek_jumlah(bersih)
        X, df_film, art = asli_bf(bersih)
        mode, demo = "real", False
    except NotImplementedError:
        bersih = demo_data.load_and_clean(path)
        _cek_jumlah(bersih)
        X, df_film, art = demo_data.build_features(bersih)
        mode, demo = "demo", True
    _tandai("load_and_clean", demo)
    _tandai("build_features", demo)
    _pasang_data(X, df_film, art, sumber, mode, ekstra=_ekstra_film(path))
    return mode


def muat_dataset_bawaan() -> str:
    """Muat ``data/raw/movies_metadata.csv``; jika belum bisa, pakai data demo.

    Return string ``"real"`` / ``"demo"`` / ``"demo-fallback"`` (CSV bawaan tidak ada).
    """
    arts = _muat_artefak_model()
    if arts is not None:
        _pasang_asli(arts)
        return "real"
    if not CSV_BAWAAN.exists():
        X, df, art, eks = _data_demo()
        _pasang_data(X, df, art, "demo sintetis (1.200 film karangan)", "demo", ekstra=eks)
        _tandai("load_and_clean", True)
        _tandai("build_features", True)
        _otomatis_clustering()
        return "demo-fallback"
    mode = _muat_dari_path(str(CSV_BAWAAN), "data/raw/movies_metadata.csv")
    return mode


def muat_dataset_unggahan(isi: bytes, nama_file: str) -> str:
    """Simpan unggahan sementara lalu proses seperti dataset bawaan."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(isi)
        path = f.name
    try:
        return _muat_dari_path(path, f"unggahan: {nama_file}")
    finally:
        Path(path).unlink(missing_ok=True)


def gunakan_data_demo() -> None:
    """Kembali ke dataset demo sintetis beserta model demo."""
    X, df, art, eks = _data_demo()
    _pasang_data(X, df, art, "demo sintetis (1.200 film karangan)", "demo", ekstra=eks)
    st.session_state.bagian_demo = set()
    _otomatis_clustering()


def _otomatis_clustering() -> None:
    """Pada data demo, langsung buat model supaya halaman Hasil/Evaluasi bisa dicoba."""
    try:
        jalankan_clustering(**DEFAULT_PARAM)
        hitung_pilih_c(DEFAULT_PARAM["m"], DEFAULT_PARAM["error"], DEFAULT_PARAM["maxiter"])
    except Exception:  # noqa: BLE001
        pass


def _kunci_film_baru(judul, rating, vote, pop, durasi, tahun, genres) -> dict:
    """Kamus film baru. Kunci mengikuti nama kolom dataset (asumsi, lihat docs)."""
    return {"title": judul.strip(), "vote_average": float(rating), "vote_count": float(vote),
            "popularity": float(pop), "runtime": float(durasi), "year": int(tahun),
            "release_date": f"{int(tahun)}-01-01", "genres": list(genres)}


def _segarkan_film_baru() -> None:
    """Hitung ulang u dan koordinat 2D film baru setelah model berubah."""
    ss = st.session_state
    fb = ss.get("new_film")
    if not fb or not model_siap():
        return
    try:
        fb["u"], demo = _hitung_u_baru(fb["x"])
        _tandai("membership", demo)
        fb["xy"] = tuple(_proyeksi(fb["x"])[0])
    except Exception:  # noqa: BLE001
        ss.new_film = None


def _ke_fcm(a: np.ndarray) -> np.ndarray:
    """Ubah fitur 14 kolom ke ruang FCM (PCA 2D Anggota 2). Tanpa pca_fcm (demo): tetap."""
    p = (st.session_state.artefak or {}).get("pca_fcm")
    if p is None:
        return a
    z = p.transform(np.atleast_2d(a))
    return z[0] if np.ndim(a) == 1 else z


def _hitung_u_baru(x: np.ndarray) -> tuple[np.ndarray, bool]:
    ss = st.session_state
    return _panggil("fcm", "membership", _ke_fcm(x), ss.fcm_result["V"], m=ss.fcm_param["m"])


def simpan_film_baru(judul, rating, vote, pop, durasi, tahun, genres) -> dict:
    """Ubah input manual menjadi film acuan baru (F-05).

    ``transform_new`` memberi vektor fitur x; ``membership`` memberi u dari V
    tersimpan (PRD 5.3). Hasil disimpan di ``session_state.new_film``.
    """
    ss = st.session_state
    if not model_siap():
        raise BackendBelumSiap("Model belum dijalankan.")
    film = _kunci_film_baru(judul, rating, vote, pop, durasi, tahun, genres)
    art = ss.artefak
    if art.get("demo"):
        x, demo = demo_data.transform_new(film, art), True
    else:
        asli = _fn_asli("preprocessing", "transform_new_data")
        if asli is None or art.get("prep") is None:
            raise BackendBelumSiap("`transform_new_data` atau preprocessing_artifacts.pkl belum tersedia.")
        baris = pd.DataFrame([{**film, "release_year": film["year"]}])
        x, demo = np.clip(asli(baris, art["prep"]).to_numpy(dtype=float)[0], 0.0, 1.0), False
    _tandai("transform_new", demo)
    ss.new_film = {"title": film["title"], "film": film, "x": x}
    _segarkan_film_baru()
    return ss.new_film


# ---------------------------------------------------------------------------
# Halaman Proses
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=16)
def _fcm_cached(_X: np.ndarray, x_token: str, c: int, m: float, error: float, maxiter: int):
    return _panggil("fcm", "fcm", _X, c, m=m, error=error, maxiter=maxiter, seed=42)


def jalankan_clustering(c: int, m: float, error: float, maxiter: int) -> dict:
    """Jalankan ``fcm.fcm`` dan simpan hasilnya (F-09). Seed terkunci = 42."""
    ss = st.session_state
    if not data_siap():
        raise BackendBelumSiap("Data belum dimuat.")
    pesan = interpret.validasi_parameter(c, m, error, maxiter)
    if pesan:
        raise ValueError(" ".join(pesan))
    Z = _ke_fcm(ss.X)
    hasil, demo = _fcm_cached(Z, _token(Z), int(c), float(m), float(error), int(maxiter))
    _tandai("fcm", demo)
    _pasang_model(hasil, {"c": int(c), "m": float(m), "error": float(error), "maxiter": int(maxiter)})
    try:
        hitung_hasil_antara()
    except Exception:  # noqa: BLE001 - hasil antara hanya pelengkap tampilan
        ss.hasil_antara = None
    return hasil


def hitung_hasil_antara(iterasi: tuple = (1, 2, 3, 5, 10)) -> list[dict] | None:
    """Hasil antara FCM: U dan V setelah iterasi ke-1, 2, 3, 5, 10 dan hasil akhir.

    Memanggil kembali ``fcm.fcm`` dengan ``maxiter = k`` dan seed yang sama (seed terkunci,
    sehingga hasilnya deterministik dan sama dengan k iterasi pertama proses penuh).
    Iterasi yang tidak lebih kecil dari jumlah iterasi akhir dilewati. Hasil disimpan di
    ``session_state.hasil_antara`` sebagai daftar dict ``{label, iter, U, V, J}``.
    """
    ss = st.session_state
    prm, akhir = ss.get("fcm_param"), ss.get("fcm_result")
    if not prm or akhir is None or akhir.get("n_iter") is None:
        ss.hasil_antara = None
        return None
    Z = _ke_fcm(ss.X)
    tok = _token(Z)
    snap = []
    for k in iterasi:
        if k >= akhir["n_iter"]:
            continue
        r, demo = _fcm_cached(Z, tok, prm["c"], prm["m"], prm["error"], int(k))
        _tandai("fcm", demo)
        J = (r.get("J_history") or [float("nan")])[-1]
        snap.append({"label": f"Iterasi {k}", "iter": int(k), "U": r["U"], "V": r["V"], "J": float(J)})
    J_akhir = (akhir.get("J_history") or [float("nan")])[-1]
    snap.append({"label": f"Akhir (iterasi {akhir['n_iter']})", "iter": int(akhir["n_iter"]),
                 "U": akhir["U"], "V": akhir["V"], "J": float(J_akhir)})
    ss.hasil_antara = snap
    return snap


def kurva_keanggotaan(fitur: int, n_titik: int = 81) -> tuple[np.ndarray, np.ndarray, float]:
    """Fungsi keanggotaan FCM terhadap satu fitur: ``u(x)`` dari ``fcm.membership``.

    Fitur lain ditahan pada rata-rata seluruh film. Return ``(grid, U_grid, nilai_rata)``.
    """
    ss = st.session_state
    X, V, m = ss.X, ss.fcm_result["V"], ss.fcm_param["m"]
    ref = X.mean(axis=0)
    grid = np.linspace(float(X[:, fitur].min()), float(X[:, fitur].max()), n_titik)
    baris, demo = [], False
    for g in grid:
        x = ref.copy()
        x[fitur] = g
        u, demo = _panggil("fcm", "membership", _ke_fcm(x), V, m=m)
        baris.append(np.asarray(u, dtype=float))
    _tandai("membership", demo)
    return grid, np.vstack(baris), float(ref[fitur])


def lokasi_kode() -> list[dict]:
    """Lokasi kode algoritma utama (live): file, nomor baris, dan sumber fungsi.

    Untuk tiap fungsi kontrak, dipakai modul asli jika sudah diimplementasikan; jika tidak,
    ditunjuk padanannya di ``src/demo_data.py`` dengan label demo. Dipakai tab Library & Kode.
    """
    daftar = [("Membersihkan data", "preprocessing", "load_and_clean"),
              ("Membentuk fitur (min-max, PCA genre)", "preprocessing", "build_features"),
              ("Validasi CSV", "preprocessing", "validate_csv"),
              ("Fuzzy C-Means (iterasi V dan U, J_m)", "fcm", "fcm"),
              ("Derajat keanggotaan film baru", "fcm", "membership"),
              ("Pemilihan jumlah cluster c", "fcm", "select_c"),
              ("KNN berbobot fuzzy (integrasi FCM → KNN)", "knn_fuzzy", "recommend"),
              ("Evaluasi Precision@K", "evaluation", "precision_at_k")]
    hasil = []
    for tahap, modul, nama in daftar:
        asli = _fn_asli(modul, nama)
        pakai, demo = asli, False
        try:
            if asli is None or "NotImplementedError" in inspect.getsource(asli):
                raise NotImplementedError
        except (NotImplementedError, OSError, TypeError):
            pakai, demo = getattr(demo_data, nama), True
        try:
            baris_kode, awal = inspect.getsourcelines(pakai)
            berkas = Path(inspect.getsourcefile(pakai)).resolve().relative_to(ROOT)
        except (OSError, TypeError, ValueError):
            baris_kode, awal, berkas = [], 0, Path("?")
        hasil.append({"tahap": tahap, "fungsi": f"{modul}.{nama}", "berkas": str(berkas).replace("\\", "/"),
                      "baris": awal, "demo": demo, "kode": "".join(baris_kode)})
    return hasil


@st.cache_data(show_spinner=False, max_entries=8)
def _select_c_cached(_X: np.ndarray, x_token: str, c_awal: int, c_akhir: int, m: float, error: float, maxiter: int):
    return _panggil("fcm", "select_c", _X, range(c_awal, c_akhir + 1), m=m, error=error, maxiter=maxiter)


def hitung_pilih_c(m: float, error: float, maxiter: int, c_awal: int = 3, c_akhir: int = 8) -> pd.DataFrame:
    """FPC dan silhouette untuk c = 3…8 lewat ``fcm.select_c`` (F-12)."""
    ss = st.session_state
    Z = _ke_fcm(ss.X)
    df, demo = _select_c_cached(Z, _token(Z), c_awal, c_akhir, float(m), float(error), int(maxiter))
    _tandai("select_c", demo)
    ss.select_c_df = df
    return df


def saran_c(df: pd.DataFrame) -> int:
    """c terbaik: peringkat rata-rata FPC dan silhouette (PRD 5.2). Hanya saran tampilan."""
    r = df["fpc"].rank(ascending=False) + df["silhouette"].rank(ascending=False)
    return int(df.loc[r.idxmin(), "c"])


@st.cache_data(show_spinner=False)
def _pca_2d_sendiri(_X: np.ndarray, x_token: str):
    return PCA(n_components=2, random_state=42).fit(_X)


def _model_2d():
    """Model proyeksi 2D: ``artefak['pca_2d']`` bila berupa objek PCA, jika tidak PCA(2) pada X."""
    ss = st.session_state
    p = (ss.artefak or {}).get("pca_2d")
    if p is not None and hasattr(p, "transform"):
        return p
    return _pca_2d_sendiri(ss.X, _token(ss.X))


def _proyeksi(a: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(a)
    if a.shape[1] == 2:          # V asli sudah berada di ruang PCA 2D
        return a
    return _model_2d().transform(a)


def koordinat_2d() -> tuple[np.ndarray, np.ndarray]:
    """(X2d, V2d): koordinat PCA 2D seluruh film dan pusat cluster (scatter F-16)."""
    ss = st.session_state
    return _proyeksi(ss.X), _proyeksi(ss.fcm_result["V"])


def nama_fitur() -> list[str]:
    """Nama kolom X untuk kurva keanggotaan; ambil dari artefak bila ada."""
    ss = st.session_state
    f = ss.X.shape[1]
    nama = (ss.artefak or {}).get("feature_names")
    if nama is not None and len(nama) == f:
        return list(nama)
    dasar = ["Rating", "Jumlah vote", "Popularitas", "Durasi", "Tahun rilis"]  # urutan PRD bagian 4
    return [dasar[i] if i < len(dasar) else f"Fitur genre {i - len(dasar) + 1}" for i in range(f)]


# ---------------------------------------------------------------------------
# Halaman Hasil
# ---------------------------------------------------------------------------

def daftar_genre() -> list[str]:
    """Genre yang muncul di dataset aktif (untuk multiselect input manual)."""
    ada = sorted({g for gs in st.session_state.film_view["_genres"] for g in gs})
    return ada or list(demo_data.GENRES)


def hover_df() -> pd.DataFrame:
    """Kolom hover scatter: title, year, genres_text (sejajar baris X)."""
    fv = st.session_state.film_view
    return pd.DataFrame({"title": fv["title"].astype(str).to_numpy(),
                         "year": fv["_year"].round().astype("Int64").astype(str).to_numpy(),
                         "genres_text": fv["_genres_text"].to_numpy()})


def indeks_film(judul: str) -> int:
    """Indeks baris pertama dengan judul tersebut."""
    return int(np.flatnonzero(st.session_state.film_view["title"].astype(str).to_numpy() == judul)[0])


def cari_rekomendasi(q_idx: int | None, k: int, alpha: float) -> dict:
    """Jalankan ``knn_fuzzy.recommend`` dan susun tabel tampilan (F-14, F-15).

    ``q_idx`` None berarti film baru dari input manual (``session_state.new_film``).
    Hasil disimpan di ``session_state.last_reco`` sebagai dict.
    """
    ss = st.session_state
    if not model_siap():
        raise BackendBelumSiap("Model belum dijalankan.")
    U = ss.fcm_result["U"]
    if q_idx is None:
        fb = ss.get("new_film")
        if not fb:
            raise ValueError("Film baru belum dibuat. Isi form di halaman Input Data lalu klik Simpan Film Baru.")
        reco, demo = _panggil("knn_fuzzy", "recommend", None, ss.X, U, k=int(k), alpha=float(alpha),
                              x_new=fb["x"], u_new=fb["u"])
        u_q, judul = fb["u"], fb["title"]
    else:
        reco, demo = _panggil("knn_fuzzy", "recommend", int(q_idx), ss.X, U, k=int(k), alpha=float(alpha))
        u_q, judul = U[int(q_idx)], str(ss.film_view.loc[int(q_idx), "title"])
    _tandai("recommend", demo)
    idx = reco["idx"].to_numpy(dtype=int)
    # pembanding KNN biasa (α = 1) dan uraian jarak untuk memperlihatkan integrasi FCM → KNN
    if q_idx is None:
        x_q = fb["x"]
        base, _ = _panggil("knn_fuzzy", "recommend", None, ss.X, U, k=int(k), alpha=1.0, x_new=fb["x"], u_new=fb["u"])
    else:
        x_q = ss.X[int(q_idx)]
        base, _ = _panggil("knn_fuzzy", "recommend", int(q_idx), ss.X, U, k=int(k), alpha=1.0)
    rincian = interpret.rincian_jarak(ss.X, U, x_q, u_q, idx, float(alpha))
    selisih = float(np.abs(rincian["d"].to_numpy() - reco["distance"].to_numpy(dtype=float)).max())
    fv = ss.film_view.iloc[idx]
    dom = U[idx].argmax(axis=1)
    tabel = pd.DataFrame({
        "No": np.arange(1, len(idx) + 1),
        "Judul": fv["title"].astype(str).to_numpy(),
        "Tahun": fv["_year"].round().astype("Int64").to_numpy(),
        "Genre": fv["_genres_text"].to_numpy(),
        "Rating": pd.to_numeric(fv["vote_average"], errors="coerce").round(1).to_numpy(),
        "Skor Kemiripan": reco["score"].to_numpy(dtype=float),
        "Cluster Dominan": [ss.nama_cluster[j] for j in dom],
        "Keanggotaan": U[idx, dom],
    })
    ss.last_reco = {"q_idx": q_idx, "judul": judul, "u_q": np.asarray(u_q, dtype=float), "idx": idx,
                    "k": int(k), "alpha": float(alpha), "tabel": tabel,
                    "rating_rata": float(np.nanmean(pd.to_numeric(fv["vote_average"], errors="coerce"))),
                    "genres": fv["_genres"].tolist(), "rincian": rincian, "selisih_jarak": selisih,
                    "idx_base": base["idx"].to_numpy(dtype=int), "ratings": pd.to_numeric(fv["vote_average"], errors="coerce").to_numpy(),
                    "judul_rec": fv["title"].astype(str).tolist()}
    return ss.last_reco


# ---------------------------------------------------------------------------
# Halaman Evaluasi
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=8)
def _precision_cached(_X, _U, _df_film, tok: str, k_list: tuple, n_sample: int, alpha: float):
    return _panggil("evaluation", "precision_at_k", _X, _U, _df_film, k_list=k_list,
                    n_sample=n_sample, alpha=alpha, seed=42)


@st.cache_data(show_spinner=False, max_entries=8)
def _silhouette_cached(_X, _U, tok: str):
    lab = _U.argmax(axis=1)
    c = _U.shape[1]
    if len(set(lab)) < 2:
        return float("nan"), np.full(c, np.nan), np.bincount(lab, minlength=c)
    ukuran = min(3000, len(_X))
    rng = np.random.default_rng(42)
    sel = rng.choice(len(_X), size=ukuran, replace=False)
    if len(set(lab[sel])) < 2:
        sel = np.arange(len(_X))
    total = float(silhouette_score(_X[sel], lab[sel]))
    s = silhouette_samples(_X[sel], lab[sel])
    per = np.array([s[lab[sel] == j].mean() if (lab[sel] == j).any() else np.nan for j in range(c)])
    return total, per, np.bincount(lab, minlength=c)


@st.cache_data(show_spinner=False, max_entries=8)
def _prediksi_cached(_X, _U, _film, tok: str, n_sample: int, alpha: float, k: int):
    """Rating aktual vs prediksi (rata-rata rating tetangga berbobot skor) untuk KNN fuzzy dan KNN biasa."""
    rng = np.random.default_rng(42)
    q = rng.choice(len(_X), size=min(n_sample, len(_X)), replace=False)
    rating = pd.to_numeric(_film["vote_average"], errors="coerce").to_numpy(dtype=float)
    pf, pb, demo = [], [], False
    for qi in q:
        for daftar, a in ((pf, alpha), (pb, 1.0)):
            r, demo = _panggil("knn_fuzzy", "recommend", int(qi), _X, _U, k=k, alpha=a)
            daftar.append(interpret.prediksi_rating(rating[r["idx"].to_numpy(dtype=int)], r["score"].to_numpy(dtype=float)))
    return q, rating[q], np.asarray(pf), np.asarray(pb), demo


@st.cache_data(show_spinner=False, max_entries=8)
def _sweep_alpha_cached(_X, _U, _df_film, tok: str, n_sample: int, alphas: tuple):
    baris, demo = [], False
    for a in alphas:
        r, demo = _panggil("evaluation", "precision_at_k", _X, _U, _df_film, k_list=(10,), n_sample=n_sample,
                           alpha=float(a), seed=42)
        from src.plots import normalisasi_tabel_precision
        r = normalisasi_tabel_precision(r)
        kolom = "baseline" if a >= 1.0 else "fuzzy"
        baris.append({"alpha": float(a), "precision": float(r[r["k"] == 10][kolom].iloc[0])})
    return pd.DataFrame(baris), demo


def jalankan_pengujian(alpha: float = 0.5, n_sample: int = 200, k_list: tuple = (5, 10, 20)) -> dict:
    """Hitung semua metrik evaluasi (F-20, F-21, brief E5).

    - FPC dari hasil FCM; silhouette (total dan per cluster) dihitung GUI dengan scikit-learn.
    - Precision@K dari ``evaluation.precision_at_k`` (fuzzy vs KNN biasa).
    - RMSE dan MAE prediksi rating: rating film acuan diprediksi dari rating tetangga (GUI).
    - Sapuan α (Precision@10 untuk α = 0, 0,25, 0,5, 0,75, 1) memperlihatkan kontribusi FCM.
    """
    ss = st.session_state
    if not model_siap():
        raise BackendBelumSiap("Model belum dijalankan.")
    X, U = ss.X, ss.fcm_result["U"]
    tok = _token(X, U, ss.data_ver)
    prec, demo = _precision_cached(X, U, ss.df_film, tok, tuple(k_list), int(n_sample), float(alpha))
    _tandai("precision_at_k", demo)
    from src.plots import normalisasi_tabel_precision
    prec = normalisasi_tabel_precision(prec)
    sil, sil_per, ukuran = _silhouette_cached(_ke_fcm(X), U, tok)
    fpc = float(ss.fcm_result.get("fpc", (U ** 2).sum() / len(U)))
    q, aktual, pf, pb, demo_p = _prediksi_cached(X, U, ss.df_film, tok, int(n_sample), float(alpha), 10)
    _tandai("recommend", demo_p)
    rmse_f, mae_f = interpret.galat_prediksi(aktual, pf)
    rmse_b, mae_b = interpret.galat_prediksi(aktual, pb)
    sweep, demo_s = _sweep_alpha_cached(X, U, ss.df_film, tok, min(int(n_sample), 100), (0.0, 0.25, 0.5, 0.75, 1.0))
    _tandai("precision_at_k", demo or demo_s)
    ss.eval_hasil = {"precision": prec, "silhouette": sil, "silhouette_per": sil_per, "ukuran": ukuran,
                     "fpc": fpc, "alpha": float(alpha), "n_sample": int(n_sample),
                     "aktual": aktual, "pred_fuzzy": pf, "pred_base": pb,
                     "judul_uji": ss.film_view["title"].astype(str).to_numpy()[q].tolist(),
                     "rmse_fuzzy": rmse_f, "mae_fuzzy": mae_f, "rmse_base": rmse_b, "mae_base": mae_b,
                     "sweep": sweep}
    return ss.eval_hasil


# ---------------------------------------------------------------------------
# Ringkasan metrik untuk kartu di halaman Hasil
# ---------------------------------------------------------------------------

def metrik_ringkas() -> dict:
    """Metrik untuk empat kartu ringkasan halaman Hasil (tidak mengubah ``eval_hasil``).

    Kunci: ``fpc``, ``silhouette``, ``precision10`` (nilai atau None bila tidak bisa dihitung),
    ``seri_fpc`` (FPC untuk c = 3…8), ``seri_sil`` (silhouette per cluster), ``seri_prec`` (Precision@K).
    Memakai ``eval_hasil`` bila Evaluasi sudah dijalankan, jika tidak dihitung ringan dan di-cache.
    """
    ss = st.session_state
    hasil = {"fpc": None, "silhouette": None, "precision10": None, "seri_fpc": [], "seri_sil": [], "seri_prec": []}
    if not model_siap():
        return hasil
    U, X = ss.fcm_result["U"], ss.X
    hasil["fpc"] = float(ss.fcm_result.get("fpc", (U ** 2).sum() / len(U)))
    tok = _token(X, U, ss.data_ver)
    ev = ss.get("eval_hasil")
    try:
        if ev:
            hasil["silhouette"], hasil["seri_sil"] = float(ev["silhouette"]), list(map(float, ev["silhouette_per"]))
        else:
            sil, per, _ = _silhouette_cached(_ke_fcm(X), U, tok)
            hasil["silhouette"], hasil["seri_sil"] = float(sil), list(map(float, per))
    except Exception:  # noqa: BLE001
        pass
    try:
        if ev:
            tab = ev["precision"]
        else:
            from src.plots import normalisasi_tabel_precision
            tab, demo = _precision_cached(X, U, ss.df_film, tok, (5, 10, 20), 60, 0.5)
            _tandai("precision_at_k", demo)
            tab = normalisasi_tabel_precision(tab)
        kol = "fuzzy" if "fuzzy" in tab.columns else "precision_fuzzy"
        baris = tab[tab["k"] == 10]
        if len(baris):
            hasil["precision10"] = float(baris.iloc[0][kol])
        hasil["seri_prec"] = list(map(float, tab[kol]))
    except Exception:  # noqa: BLE001
        pass
    sc = ss.get("select_c_df")
    if sc is not None and "fpc" in sc.columns:
        hasil["seri_fpc"] = list(map(float, sc["fpc"]))
    return hasil
