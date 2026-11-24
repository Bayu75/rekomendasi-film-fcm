"""Semua grafik FuzzyFlix (Plotly). Anggota 4: GUI & Visualisasi.

Aturan (design-system.md bagian 6):
- Setiap fungsi *murni*: data masuk, objek ``plotly.graph_objects.Figure``
  keluar. Tidak ada pemanggilan ``st.*`` dan tidak ada pembacaan file.
- Template Plotly mengikuti tema GUI (Cinematic Dark: ``plotly_dark``) dengan latar transparan.
- Warna cluster selalu dari ``CLUSTER_COLORS`` (cluster ke-n = warna yang sama
  di semua grafik). Jika c > 8, warna diulang dari awal (modulo 8).
- Tinggi standar 420 px, grafik utama (scatter) 520 px.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Konstanta tampilan (design-system.md bagian 2)
# ---------------------------------------------------------------------------

CLUSTER_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756",
                  "#72B7B2", "#EECA3B", "#B279A2", "#9D755D"]

PRIMARY = "#E50914"        # aksen / outline film rekomendasi dan film acuan
BASELINE_COLOR = "#A0A6B1"  # KNN biasa
TEXT_MUTED = "#A0A6B1"
TEXT_COLOR = "#F5F5F7"
GRID_COLOR = "#232838"
BG_COLOR = "#0B0D12"
SURFACE_COLOR = "#151821"
BORDER_COLOR = "#2A2F3A"
PLOTLY_TEMPLATE = "plotly_dark"
FONT_FAMILY = "Inter, Source Sans 3, Segoe UI, sans-serif"
# Heatmap keanggotaan pada latar gelap: 0 = gelap menyatu dengan kartu, 1 = biru terang.
HEATMAP_SCALE = [[0.0, "#151821"], [0.45, "#1E3A8A"], [0.75, "#3B82F6"], [1.0, "#DBEAFE"]]
HEIGHT = 420
HEIGHT_MAIN = 520

# Fitur numerik dasar pada X (urutan mengikuti PRD bagian 4).
FEATURE_NAMES_DEFAULT = ["Rating", "Jumlah vote", "Popularitas", "Durasi", "Tahun rilis"]


# ---------------------------------------------------------------------------
# Helper internal
# ---------------------------------------------------------------------------

def cluster_color(j: int) -> str:
    """Warna cluster ke-j (indeks dari 0). Diulang jika j >= 8."""
    return CLUSTER_COLORS[int(j) % len(CLUSTER_COLORS)]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Ubah ``#RRGGBB`` menjadi string ``rgba(r,g,b,a)``."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _default_names(c: int) -> list[str]:
    return [f"C{j + 1}" for j in range(c)]


def _layout(fig: go.Figure, title: str | None = None, xtitle: str | None = None,
            ytitle: str | None = None, height: int = HEIGHT, legend: bool = True) -> go.Figure:
    """Terapkan gaya seragam: template tema, latar transparan, judul sumbu."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16)) if title else None,
        margin=dict(l=60, r=20, t=60 if title else 20, b=55),
        showlegend=legend,
        legend=dict(orientation="v", bgcolor="rgba(0,0,0,0)"),
    )
    if xtitle is not None:
        fig.update_xaxes(title_text=xtitle)
    if ytitle is not None:
        fig.update_yaxes(title_text=ytitle)
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


def _empty_figure(pesan: str, title: str | None = None, height: int = HEIGHT) -> go.Figure:
    """Figure kosong berisi satu kalimat penjelasan (data belum tersedia)."""
    fig = go.Figure()
    fig.add_annotation(text=pesan, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color=TEXT_MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title=title, height=height, legend=False)


# ---------------------------------------------------------------------------
# Halaman Proses
# ---------------------------------------------------------------------------

def plot_objective(J_history: Sequence[float], log_y: bool = False) -> go.Figure:
    """Grafik fungsi objektif J_m per iterasi (F-11).

    Parameter
    ----------
    J_history : urutan float
        Nilai J_m tiap iterasi (``fcm(...)['J_history']``).
    log_y : bool
        Skala logaritmik pada sumbu Y bila penurunan J sangat tajam.

    Return
    ------
    go.Figure  (line; judul sumbu "Iterasi" dan "Fungsi objektif J_m").
    """
    J = np.asarray(list(J_history), dtype=float)
    if J.size == 0:
        return _empty_figure("Riwayat J_m belum tersedia. Klik Jalankan Clustering "
                             "untuk melihat konvergensi.", "Konvergensi fungsi objektif")
    it = np.arange(1, J.size + 1)
    fig = go.Figure(go.Scatter(
        x=it, y=J, mode="lines+markers" if J.size <= 60 else "lines",
        line=dict(color=PRIMARY, width=2.5), marker=dict(size=6),
        name="J_m", hovertemplate="Iterasi %{x}<br>J_m = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[it[-1]], y=[J[-1]], mode="markers+text", text=[f"{J[-1]:.2f}"],
        textposition="top right", marker=dict(size=11, color="#FFFFFF",
                                              line=dict(color=PRIMARY, width=2)),
        showlegend=False, hoverinfo="skip"))
    _layout(fig, "Konvergensi fungsi objektif", "Iterasi", "Fungsi objektif J_m", legend=False)
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def plot_select_c(df: pd.DataFrame, c_selected: int | None = None) -> go.Figure:
    """FPC dan silhouette untuk tiap jumlah cluster c (F-12).

    Parameter
    ----------
    df : DataFrame berkolom ``c, fpc, silhouette`` (keluaran ``select_c``).
    c_selected : int, opsional
        Nilai c terpilih; titiknya ditandai lingkaran besar dan garis putus.
    """
    d = df.sort_values("c")
    c = d["c"].to_numpy()
    fig = go.Figure()
    for kolom, label, warna in (("fpc", "FPC", "#4C78A8"), ("silhouette", "Silhouette", "#F58518")):
        fig.add_trace(go.Scatter(
            x=c, y=d[kolom], mode="lines+markers", name=label,
            line=dict(color=warna, width=2.5), marker=dict(size=8),
            hovertemplate=f"c = %{{x}}<br>{label} = %{{y:.3f}}<extra></extra>"))
    if c_selected is not None and int(c_selected) in set(int(v) for v in c):
        row = d[d["c"] == int(c_selected)].iloc[0]
        fig.add_vline(x=int(c_selected), line=dict(color=PRIMARY, dash="dash", width=1.5))
        fig.add_trace(go.Scatter(
            x=[int(c_selected)] * 2, y=[row["fpc"], row["silhouette"]], mode="markers",
            marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=PRIMARY, width=3)),
            name=f"c terpilih = {int(c_selected)}", hoverinfo="skip"))
    _layout(fig, "FPC dan silhouette per jumlah cluster", "Jumlah cluster (c)", "Nilai metrik")
    fig.update_xaxes(dtick=1)
    fig.update_layout(legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    return fig


def plot_membership_heatmap(U: np.ndarray, titles: Sequence[str], n: int = 20,
                            cluster_names: Sequence[str] | None = None,
                            seed: int = 42) -> go.Figure:
    """Heatmap matriks keanggotaan untuk sampel n film (F-13), skala ``Blues``.

    Parameter
    ----------
    U : ndarray (n_film, c)   Matriks derajat keanggotaan.
    titles : urutan str       Judul film, sejajar dengan baris U.
    n : int                   Jumlah film sampel (diambil acak, seed tetap).
    cluster_names : opsional  Nama kolom; default C1..Cc.
    """
    U = np.asarray(U, dtype=float)
    n = int(min(max(n, 1), U.shape[0]))
    idx = np.sort(np.random.default_rng(seed).choice(U.shape[0], size=n, replace=False))
    names = list(cluster_names) if cluster_names is not None else _default_names(U.shape[1])
    kolom = [f"C{j + 1}" for j in range(U.shape[1])]
    judul = [str(titles[i]) for i in idx]
    fig = go.Figure(go.Heatmap(
        z=U[idx], x=kolom, y=judul, colorscale=HEATMAP_SCALE, zmin=0, zmax=1,
        text=np.round(U[idx], 3), texttemplate="%{text:.3f}" if U.shape[1] <= 10 else None,
        customdata=np.tile(np.asarray(names, dtype=object), (n, 1)),
        hovertemplate="%{y}<br>%{customdata}<br>Derajat keanggotaan = %{z:.3f}<extra></extra>",
        colorbar=dict(title="Derajat<br>keanggotaan")))
    _layout(fig, f"Matriks keanggotaan ({n} film sampel)", "Cluster", "Film",
            height=max(HEIGHT, 22 * n + 140), legend=False)
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def plot_pca_variance(pca) -> go.Figure:
    """Explained variance PCA genre: bar per komponen + garis kumulatif.

    ``pca`` boleh objek sklearn ``PCA`` (punya ``explained_variance_ratio_``)
    atau array rasio variansi per komponen.
    """
    rasio = np.asarray(getattr(pca, "explained_variance_ratio_", pca), dtype=float)
    komp = [f"PC{i + 1}" for i in range(rasio.size)]
    kum = np.cumsum(rasio)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=komp, y=rasio, name="Variansi per komponen",
                         marker_color="#4C78A8",
                         hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=komp, y=kum, mode="lines+markers", name="Kumulatif",
                             line=dict(color=PRIMARY, width=2.5),
                             hovertemplate="%{x}<br>kumulatif %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.8, line=dict(color=BASELINE_COLOR, dash="dot"),
                  annotation_text="80%", annotation_position="bottom right")
    _layout(fig, "Explained variance PCA genre", "Komponen utama", "Rasio variansi")
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    return fig


def plot_membership_curve(X: np.ndarray, U: np.ndarray, feature: int,
                          feature_label: str | None = None, n_bins: int = 20,
                          cluster_names: Sequence[str] | None = None) -> go.Figure:
    """Kurva keanggotaan terhadap satu fitur (F-25, P2).

    Rentang nilai fitur dibagi ``n_bins`` interval berisi jumlah film hampir
    sama (kuantil); tiap titik adalah rata-rata keanggotaan film dalam interval.

    Parameter
    ----------
    X : ndarray (n, f)   Matriks fitur.
    U : ndarray (n, c)   Matriks keanggotaan.
    feature : int        Indeks kolom X yang dipakai sebagai sumbu X.
    feature_label : str  Nama fitur untuk judul sumbu (default ``Fitur {i+1}``).
    """
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    x = X[:, int(feature)]
    label = feature_label or f"Fitur {int(feature) + 1}"
    names = list(cluster_names) if cluster_names is not None else _default_names(U.shape[1])
    # kuantil -> tiap bin punya jumlah titik kurang lebih sama
    tepi = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    bin_id = np.clip(np.searchsorted(tepi, x, side="right") - 1, 0, max(len(tepi) - 2, 0))
    fig = go.Figure()
    for j in range(U.shape[1]):
        xs, ys = [], []
        for b in range(max(len(tepi) - 1, 1)):
            mask = bin_id == b
            if mask.any():
                xs.append(float(x[mask].mean()))
                ys.append(float(U[mask, j].mean()))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=names[j],
            line=dict(color=cluster_color(j), width=2.5, shape="spline", smoothing=0.6),
            marker=dict(size=6),
            hovertemplate=f"{label} = %{{x:.3f}}<br>keanggotaan = %{{y:.3f}}<extra>{names[j]}</extra>"))
    _layout(fig, f"Keanggotaan cluster terhadap {label}",
            f"{label} (ternormalisasi 0–1)", "Rata-rata derajat keanggotaan")
    fig.update_yaxes(range=[0, 1])
    return fig


# ---------------------------------------------------------------------------
# Halaman Hasil
# ---------------------------------------------------------------------------

def _hover_text(U: np.ndarray, hover: pd.DataFrame | None) -> list[str]:
    """Teks hover tiap titik: judul, tahun, genre, keanggotaan tiap cluster."""
    n, c = U.shape
    u_txt = ["<br>".join(f"C{j + 1}: {U[i, j]:.3f}" for j in range(c)) for i in range(n)]
    if hover is None:
        return [f"Film #{i}<br>{u_txt[i]}" for i in range(n)]
    judul = hover["title"].astype(str).tolist() if "title" in hover else [f"Film #{i}" for i in range(n)]
    tahun = hover["year"].tolist() if "year" in hover else [""] * n
    genre = hover["genres_text"].astype(str).tolist() if "genres_text" in hover else [""] * n
    return [f"<b>{judul[i]}</b><br>{tahun[i]} · {genre[i]}<br>{u_txt[i]}" for i in range(n)]


def plot_clusters(X2d: np.ndarray, U: np.ndarray, V2d: np.ndarray,
                  q_idx: int | None = None, rec_idx: Sequence[int] | None = None,
                  names: Sequence[str] | None = None, hover: pd.DataFrame | None = None,
                  query_xy: Sequence[float] | None = None,
                  query_label: str = "Film acuan") -> go.Figure:
    """Scatter cluster 2D, grafik utama halaman Hasil (F-16).

    Titik berwarna cluster dominan; opacity = 0.25 + 0.75 × keanggotaan maks.
    Urutan lapisan: titik film → pusat cluster → film rekomendasi → film acuan.

    Parameter
    ----------
    X2d : ndarray (n, 2)   Koordinat PCA 2D seluruh film.
    U : ndarray (n, c)     Matriks keanggotaan.
    V2d : ndarray (c, 2)   Pusat cluster pada ruang PCA 2D yang sama.
    q_idx : int | None     Indeks film acuan dari dataset (None jika film baru).
    rec_idx : urutan int   Indeks film rekomendasi.
    names : urutan str     Nama cluster untuk legend.
    hover : DataFrame      Kolom opsional ``title, year, genres_text`` (sejajar X2d).
    query_xy : (x, y)      Koordinat film acuan bila ``q_idx`` None (film input manual).
    """
    X2d = np.asarray(X2d, dtype=float)
    U = np.asarray(U, dtype=float)
    V2d = np.asarray(V2d, dtype=float)
    n, c = U.shape
    names = list(names) if names is not None else _default_names(c)
    dominan = U.argmax(axis=1)
    umax = U.max(axis=1)
    opacity = 0.25 + 0.75 * umax
    teks = np.asarray(_hover_text(U, hover), dtype=object)
    Trace = go.Scattergl if n > 3000 else go.Scatter

    fig = go.Figure()
    for j in range(c):  # 1) titik film, satu trace per cluster (legend)
        m = dominan == j
        fig.add_trace(Trace(
            x=X2d[m, 0], y=X2d[m, 1], mode="markers", name=names[j],
            marker=dict(color=cluster_color(j), size=7, opacity=opacity[m], line=dict(width=0)),
            text=teks[m], hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(  # 2) pusat cluster
        x=V2d[:, 0], y=V2d[:, 1], mode="markers+text", name="Pusat cluster",
        text=[f"C{j + 1}" for j in range(c)], textposition="top center",
        textfont=dict(size=11, color=TEXT_COLOR),
        marker=dict(symbol="x", size=16, color=[cluster_color(j) for j in range(c)],
                    line=dict(width=3, color=[cluster_color(j) for j in range(c)])),
        hovertemplate="Pusat %{text}<extra></extra>"))
    if rec_idx is not None and len(rec_idx) > 0:  # 3) film rekomendasi
        r = np.asarray(list(rec_idx), dtype=int)
        fig.add_trace(go.Scatter(
            x=X2d[r, 0], y=X2d[r, 1], mode="markers", name="Rekomendasi",
            marker=dict(symbol="circle-open", size=12, line=dict(color=PRIMARY, width=2.5)),
            text=teks[r], hovertemplate="%{text}<extra>Rekomendasi</extra>"))
    qx = qy = None  # 4) film acuan (paling atas)
    if q_idx is not None:
        qx, qy = X2d[int(q_idx), 0], X2d[int(q_idx), 1]
        q_text = teks[int(q_idx)]
    elif query_xy is not None:
        qx, qy = float(query_xy[0]), float(query_xy[1])
        q_text = f"<b>{query_label}</b>"
    if qx is not None:
        fig.add_trace(go.Scatter(
            x=[qx], y=[qy], mode="markers", name=query_label,
            marker=dict(symbol="star", size=18, color="#FFFFFF", line=dict(color=PRIMARY, width=2)),
            text=[q_text], hovertemplate="%{text}<extra>" + query_label + "</extra>"))
    _layout(fig, "Peta cluster film (PCA 2D)", "Komponen PCA 1", "Komponen PCA 2", height=HEIGHT_MAIN)
    fig.update_layout(legend=dict(orientation="h", y=-0.15, x=0), margin=dict(l=60, r=20, t=60, b=90))
    return fig


def plot_query_membership(u_q: np.ndarray, names: Sequence[str] | None = None,
                          title: str = "Derajat keanggotaan film acuan") -> go.Figure:
    """Bar horizontal derajat keanggotaan film acuan (F-17).

    Warna batang = warna cluster. ``u_q`` berbentuk (c,); ``names`` nama cluster.
    """
    u = np.asarray(u_q, dtype=float).ravel()
    names = list(names) if names is not None else _default_names(u.size)
    fig = go.Figure(go.Bar(
        x=u, y=names, orientation="h",
        marker_color=[cluster_color(j) for j in range(u.size)],
        text=[f"{v:.3f}" for v in u], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>derajat keanggotaan = %{x:.3f}<extra></extra>"))
    _layout(fig, title, "Derajat keanggotaan (0–1)", "Cluster", legend=False)
    fig.update_xaxes(range=[0, 1.12])
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def plot_cluster_radar(V_original: pd.DataFrame | np.ndarray, names: Sequence[str] | None = None,
                       feature_names: Sequence[str] | None = None,
                       ref: pd.DataFrame | None = None) -> go.Figure:
    """Radar ciri pusat cluster (F-19, P1).

    Parameter
    ----------
    V_original : DataFrame/ndarray (c, f_baca)
        Pusat cluster pada satuan asli yang mudah dibaca (rating, vote,
        popularitas, durasi, tahun). Jika DataFrame, nama kolom dipakai sebagai
        label sumbu.
    names : urutan str      Nama cluster.
    feature_names : urutan  Label sumbu bila ``V_original`` berupa ndarray.
    ref : DataFrame, opsional
        Data seluruh film (kolom sama) untuk skala 0–1 global tiap sumbu. Tanpa
        ``ref``, skala adalah min–max antar pusat cluster (perbedaan relatif).
    """
    if isinstance(V_original, pd.DataFrame):
        V = V_original.copy()
    else:
        arr = np.asarray(V_original, dtype=float)
        cols = list(feature_names) if feature_names is not None else FEATURE_NAMES_DEFAULT[:arr.shape[1]]
        V = pd.DataFrame(arr, columns=cols)
    c = len(V)
    names = list(names) if names is not None else _default_names(c)
    ref_df = ref if ref is not None else V
    lo = ref_df[V.columns].min()
    hi = ref_df[V.columns].max()
    rentang = (hi - lo).replace(0, np.nan)
    norm = ((V - lo) / rentang).fillna(0.5).clip(0, 1)
    axes = list(V.columns)
    fig = go.Figure()
    for j in range(c):
        r = norm.iloc[j].tolist()
        asli = V.iloc[j].tolist()
        fig.add_trace(go.Scatterpolar(
            r=r + r[:1], theta=axes + axes[:1], name=names[j], fill="toself",
            line=dict(color=cluster_color(j), width=2),
            fillcolor=hex_to_rgba(cluster_color(j), 0.15),
            customdata=asli + asli[:1],
            hovertemplate="%{theta}<br>nilai asli = %{customdata:.2f}<br>skala 0–1 = %{r:.2f}"
                          f"<extra>{names[j]}</extra>"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 1], gridcolor=GRID_COLOR, tickfont=dict(size=10)),
                                 angularaxis=dict(gridcolor=GRID_COLOR)))
    _layout(fig, "Ciri pusat cluster (skala 0–1)", height=HEIGHT_MAIN)
    return fig


# ---------------------------------------------------------------------------
# Halaman Evaluasi
# ---------------------------------------------------------------------------

_ALIAS_FUZZY = ("precision_fuzzy", "fuzzy", "p_fuzzy", "knn_fuzzy")
_ALIAS_BASE = ("precision_baseline", "baseline", "p_baseline", "knn_biasa", "precision_knn")


def _pilih_kolom(df: pd.DataFrame, kandidat: Sequence[str], peran: str) -> str:
    for k in kandidat:
        if k in df.columns:
            return k
    raise ValueError(f"Kolom Precision@K untuk {peran} tidak ditemukan. "
                     f"Kolom yang tersedia: {', '.join(map(str, df.columns))}. "
                     f"Diharapkan salah satu dari: {', '.join(kandidat)}.")


def normalisasi_tabel_precision(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan nama kolom keluaran ``precision_at_k`` menjadi ``k, fuzzy, baseline``.

    Kontrak PRD hanya menyebut "DataFrame", jadi beberapa nama kolom umum
    diterima (lihat ``_ALIAS_FUZZY`` dan ``_ALIAS_BASE``).
    """
    kf = _pilih_kolom(df, _ALIAS_FUZZY, "KNN fuzzy")
    kb = _pilih_kolom(df, _ALIAS_BASE, "KNN biasa")
    kk = "k" if "k" in df.columns else df.columns[0]
    out = pd.DataFrame({"k": df[kk].astype(int), "fuzzy": df[kf].astype(float),
                        "baseline": df[kb].astype(float)})
    return out.sort_values("k").reset_index(drop=True)


def plot_precision(df: pd.DataFrame) -> go.Figure:
    """Grouped bar Precision@K: KNN fuzzy vs KNN biasa (F-21).

    ``df`` = keluaran ``precision_at_k`` (kolom k, precision_fuzzy,
    precision_baseline; alias umum diterima). Baseline berwarna abu-abu
    ``#A0A6B1`` dengan pola garis.
    """
    d = normalisasi_tabel_precision(df)
    kat = [f"Precision@{k}" for k in d["k"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=kat, y=d["fuzzy"], name="KNN fuzzy (FCM + KNN)", marker_color=PRIMARY,
                         text=[f"{v:.3f}" for v in d["fuzzy"]], textposition="outside",
                         hovertemplate="%{x}<br>KNN fuzzy = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=kat, y=d["baseline"], name="KNN biasa (α = 1)",
                         marker=dict(color=BASELINE_COLOR, pattern=dict(shape="/", fgcolor=BG_COLOR)),
                         text=[f"{v:.3f}" for v in d["baseline"]], textposition="outside",
                         hovertemplate="%{x}<br>KNN biasa = %{y:.3f}<extra></extra>"))
    _layout(fig, "Precision@K: KNN fuzzy vs KNN biasa", "Nilai K", "Precision (0–1)")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    fig.update_yaxes(range=[0, 1.1])
    return fig


def plot_cluster_quality(sizes: Sequence[int] | np.ndarray,
                         silhouette_per_cluster: Sequence[float] | np.ndarray | None = None,
                         names: Sequence[str] | None = None) -> go.Figure:
    """Ukuran cluster (jumlah film dominan) dan silhouette per cluster.

    Dua panel berdampingan; warna batang = warna cluster. Jika
    ``silhouette_per_cluster`` None, hanya panel ukuran yang ditampilkan.
    """
    sizes = np.asarray(sizes, dtype=float)
    c = sizes.size
    names = list(names) if names is not None else _default_names(c)
    warna = [cluster_color(j) for j in range(c)]
    ada_sil = silhouette_per_cluster is not None
    fig = make_subplots(rows=1, cols=2 if ada_sil else 1,
                        subplot_titles=("Ukuran cluster", "Silhouette per cluster") if ada_sil
                        else ("Ukuran cluster",))
    fig.add_trace(go.Bar(x=[f"C{j + 1}" for j in range(c)], y=sizes, marker_color=warna,
                         customdata=names, text=[f"{int(s):,}".replace(",", ".") for s in sizes],
                         textposition="outside", cliponaxis=False,
                         hovertemplate="%{customdata}<br>%{y:.0f} film<extra></extra>"), row=1, col=1)
    fig.update_xaxes(title_text="Cluster", row=1, col=1)
    fig.update_yaxes(title_text="Jumlah film (cluster dominan)", row=1, col=1)
    if ada_sil:
        sil = np.asarray(silhouette_per_cluster, dtype=float)
        fig.add_trace(go.Bar(x=[f"C{j + 1}" for j in range(c)], y=sil, marker_color=warna,
                             customdata=names, text=[f"{v:.3f}" for v in sil],
                             textposition="outside", cliponaxis=False,
                             hovertemplate="%{customdata}<br>silhouette = %{y:.3f}<extra></extra>"),
                      row=1, col=2)
        fig.add_hline(y=0.25, line=dict(color=BASELINE_COLOR, dash="dot"), row=1, col=2,
                      annotation_text="target 0,25", annotation_position="top left")
        fig.update_xaxes(title_text="Cluster", row=1, col=2)
        fig.update_yaxes(title_text="Rata-rata silhouette", row=1, col=2)
    _layout(fig, None, legend=False)
    fig.update_layout(margin=dict(l=60, r=20, t=40, b=55))
    return fig


# ---------------------------------------------------------------------------
# Halaman Informasi
# ---------------------------------------------------------------------------

def _tema_dot(dot: str) -> str:
    """Isi penanda warna pada string DOT dengan token tema aktif."""
    for kunci, nilai in (("__BORDER__", BORDER_COLOR), ("__SURFACE__", SURFACE_COLOR), ("__TEXT__", TEXT_COLOR),
                         ("__MUTED__", TEXT_MUTED), ("__PRIMARY__", PRIMARY)):
        dot = dot.replace(kunci, nilai)
    return dot



def arsitektur_dot() -> str:
    """Diagram arsitektur sistem (F-24) dalam format Graphviz DOT.

    Dikembalikan sebagai string agar bisa ditampilkan ``st.graphviz_chart``
    maupun dipakai untuk gambar laporan.
    """
    return _tema_dot("""
digraph G {
  rankdir=TB; bgcolor="transparent"; nodesep=0.25; ranksep=0.25; size="7,9";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10,
        color="__BORDER__", fillcolor="__SURFACE__", fontcolor="__TEXT__"];
  edge [color="__MUTED__", fontcolor="__MUTED__", fontname="Helvetica", fontsize=10];
  data [label="Dataset film\\n(CSV bawaan / unggahan)"];
  pre  [label="Preprocessing\\nfilter · fitur · min-max"];
  pca  [label="PCA genre"];
  fcm  [label="Fuzzy C-Means\\nV dan U", fillcolor="__PRIMARY__"];
  model[label="Model tersimpan\\nmodels/", shape=cylinder];
  in   [label="Input pengguna\\nfilm acuan · k · α"];
  knn  [label="KNN berbobot fuzzy\\nd = α‖x‖ + (1−α)‖u‖", fillcolor="__PRIMARY__"];
  rec  [label="Top-k rekomendasi\\n+ interpretasi"];
  ev   [label="Evaluasi\\nFPC · silhouette · Precision@K"];
  gui  [label="GUI Streamlit\\n6 halaman + Plotly", fillcolor="#2563EB"];
  data -> pre -> pca -> fcm -> model;
  in -> knn; model -> knn -> rec -> gui;
  fcm -> ev; knn -> ev; ev -> gui;
}
""")


# ---------------------------------------------------------------------------
# Tambahan sesuai ketentuan tugas (integrasi FCM→KNN, hasil antara, aktual vs prediksi)
# ---------------------------------------------------------------------------

FITUR_COLOR = "#A0A6B1"  # bagian jarak fitur pada jarak KNN fuzzy


def plot_distance_breakdown(rincian: pd.DataFrame, titles: Sequence[str], alpha: float) -> go.Figure:
    """Bar bertumpuk: uraian jarak KNN fuzzy tiap rekomendasi (integrasi FCM → KNN).

    Segmen abu-abu = ``α·‖x_q − x_i‖`` (jarak fitur), segmen merah =
    ``(1−α)·‖u_q − u_i‖`` (jarak derajat keanggotaan dari FCM). Panjang total = jarak ``d``.

    Parameter
    ----------
    rincian : DataFrame berkolom ``bagian_fitur, bagian_mem, d`` (``interpret.rincian_jarak``).
    titles : judul film rekomendasi, sejajar dengan baris ``rincian``.
    alpha : α yang dipakai (untuk judul grafik).
    """
    judul = [f"{i + 1}. {t}" for i, t in enumerate(titles)]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=judul, x=rincian["bagian_fitur"], orientation="h", name=f"α·jarak fitur (α = {alpha:.1f})",
                         marker_color=FITUR_COLOR,
                         hovertemplate="%{y}<br>α·‖x_q − x_i‖ = %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(y=judul, x=rincian["bagian_mem"], orientation="h",
                         name=f"(1−α)·jarak keanggotaan FCM ({1 - alpha:.1f})", marker_color=PRIMARY,
                         hovertemplate="%{y}<br>(1−α)·‖u_q − u_i‖ = %{x:.3f}<extra></extra>"))
    _layout(fig, "Uraian jarak KNN fuzzy: fitur + keanggotaan FCM", "Jarak gabungan d", "Film rekomendasi",
            height=max(HEIGHT, 30 * len(judul) + 170))
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.25, yanchor="top", x=0), margin=dict(l=60, r=20, t=60, b=110))
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def plot_membership_compare(U_rows: np.ndarray, row_labels: Sequence[str],
                            cluster_names: Sequence[str] | None = None) -> go.Figure:
    """Heatmap vektor keanggotaan film acuan dan film rekomendasi (tanpa sampling).

    Baris pertama biasanya film acuan. Dua baris dengan pola warna mirip = jarak
    keanggotaan ``‖u_q − u_i‖`` kecil, sehingga film itu naik peringkat pada KNN fuzzy.
    """
    U_rows = np.asarray(U_rows, dtype=float)
    names = list(cluster_names) if cluster_names is not None else _default_names(U_rows.shape[1])
    fig = go.Figure(go.Heatmap(
        z=U_rows, x=[f"C{j + 1}" for j in range(U_rows.shape[1])], y=list(row_labels), colorscale=HEATMAP_SCALE,
        zmin=0, zmax=1, text=np.round(U_rows, 3), texttemplate="%{text:.3f}",
        customdata=np.tile(np.asarray(names, dtype=object), (U_rows.shape[0], 1)),
        hovertemplate="%{y}<br>%{customdata}<br>u = %{z:.3f}<extra></extra>", colorbar=dict(title="u")))
    _layout(fig, "Vektor keanggotaan u: film acuan vs rekomendasi", "Cluster", "Film",
            height=max(HEIGHT, 28 * U_rows.shape[0] + 140), legend=False)
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def plot_rating_comparison(rating_acuan: float, titles: Sequence[str], ratings: Sequence[float],
                           judul_acuan: str = "Film acuan") -> go.Figure:
    """Bar rating tiap film rekomendasi dengan garis rating film acuan (perbandingan aktual)."""
    judul = [f"{i + 1}. {t}" for i, t in enumerate(titles)]
    fig = go.Figure(go.Bar(x=judul, y=list(ratings), marker_color="#4C78A8",
                           hovertemplate="%{x}<br>rating = %{y:.1f}<extra></extra>", name="Rekomendasi"))
    if rating_acuan is not None and np.isfinite(rating_acuan):
        fig.add_hline(y=rating_acuan, line=dict(color=PRIMARY, dash="dash", width=2),
                      annotation_text=f"{judul_acuan}: {rating_acuan:.1f}", annotation_position="top left")
    _layout(fig, "Rating film acuan vs film rekomendasi", "Film rekomendasi", "Rating (0–10)", legend=False)
    fig.update_yaxes(range=[0, 10])
    fig.update_xaxes(showticklabels=False)
    return fig


def plot_actual_vs_pred(actual: Sequence[float], pred_fuzzy: Sequence[float], pred_base: Sequence[float],
                        titles: Sequence[str] | None = None) -> go.Figure:
    """Rating aktual vs rating prediksi sistem (KNN fuzzy dan KNN biasa).

    Tiap titik adalah satu film acuan uji. Garis putus-putus y = x adalah prediksi
    sempurna; makin dekat titik ke garis, makin baik prediksinya.
    """
    a = np.asarray(actual, dtype=float)
    teks = list(titles) if titles is not None else [f"Film #{i}" for i in range(len(a))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a, y=pred_base, mode="markers", name="KNN biasa (α = 1)", text=teks,
                             marker=dict(color=BASELINE_COLOR, size=7, opacity=0.6, symbol="diamond"),
                             hovertemplate="%{text}<br>aktual %{x:.1f} · prediksi %{y:.2f}<extra>KNN biasa</extra>"))
    fig.add_trace(go.Scatter(x=a, y=pred_fuzzy, mode="markers", name="KNN fuzzy", text=teks,
                             marker=dict(color=PRIMARY, size=7, opacity=0.7),
                             hovertemplate="%{text}<br>aktual %{x:.1f} · prediksi %{y:.2f}<extra>KNN fuzzy</extra>"))
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Prediksi sempurna (y = x)",
                             line=dict(color=TEXT_COLOR, dash="dash", width=1.5), hoverinfo="skip"))
    _layout(fig, "Rating aktual vs prediksi sistem", "Rating aktual (vote_average)",
            "Prediksi dari tetangga (rata-rata berbobot skor)")
    fig.update_layout(legend=dict(orientation="h", y=-0.22, yanchor="top", x=0), margin=dict(l=60, r=20, t=60, b=110))
    return fig


def plot_alpha_sweep(df: pd.DataFrame) -> go.Figure:
    """Precision@10 untuk beberapa nilai α: bukti kontribusi derajat keanggotaan FCM.

    ``df`` berkolom ``alpha, precision``. Titik α = 1 adalah KNN biasa (baseline, abu-abu).
    """
    d = df.sort_values("alpha")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["alpha"], y=d["precision"], mode="lines+markers", name="KNN fuzzy",
                             line=dict(color=PRIMARY, width=2.5), marker=dict(size=9),
                             hovertemplate="α = %{x:.2f}<br>Precision@10 = %{y:.3f}<extra></extra>"))
    base = d[d["alpha"] >= 1.0]
    if len(base):
        fig.add_trace(go.Scatter(x=base["alpha"], y=base["precision"], mode="markers", name="KNN biasa (α = 1)",
                                 marker=dict(size=14, color=BASELINE_COLOR, line=dict(color=TEXT_COLOR, width=2)),
                                 hoverinfo="skip"))
        fig.add_hline(y=float(base["precision"].iloc[0]), line=dict(color=BASELINE_COLOR, dash="dot"))
    _layout(fig, "Precision@10 menurut nilai α", "α (bobot jarak fitur; 0 = hanya keanggotaan, 1 = KNN biasa)",
            "Precision@10")
    fig.update_layout(legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    return fig


def plot_center_trajectory(X2d: np.ndarray, V2d_list: Sequence[np.ndarray], labels: Sequence[str],
                           U: np.ndarray, names: Sequence[str] | None = None) -> go.Figure:
    """Perpindahan pusat cluster antar iterasi pada ruang PCA 2D (hasil antara FCM).

    ``V2d_list`` = pusat tiap snapshot (c, 2) berurutan dari iterasi awal sampai akhir;
    ``labels`` = keterangan snapshot (mis. "Iterasi 1", "Akhir"). Titik film diberi
    warna cluster dominan; garis menghubungkan posisi pusat tiap cluster.
    """
    X2d = np.asarray(X2d, dtype=float)
    U = np.asarray(U, dtype=float)
    c = U.shape[1]
    names = list(names) if names is not None else _default_names(c)
    dom = U.argmax(axis=1)
    fig = go.Figure()
    for j in range(c):
        m = dom == j
        fig.add_trace(go.Scatter(x=X2d[m, 0], y=X2d[m, 1], mode="markers", name=names[j], showlegend=False,
                                 marker=dict(color=cluster_color(j), size=5, opacity=0.25), hoverinfo="skip"))
    for j in range(c):
        xs = [v[j, 0] for v in V2d_list]
        ys = [v[j, 1] for v in V2d_list]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers+text", name=f"Pusat {names[j]}",
                                 text=[""] * (len(xs) - 1) + [f"C{j + 1}"], textposition="top center",
                                 line=dict(color=cluster_color(j), width=2),
                                 marker=dict(color=cluster_color(j), size=[7] * (len(xs) - 1) + [16],
                                             symbol=["circle"] * (len(xs) - 1) + ["x"],
                                             line=dict(width=3, color=cluster_color(j))),
                                 customdata=labels, hovertemplate="%{customdata}<extra>" + names[j] + "</extra>"))
    _layout(fig, "Perpindahan pusat cluster per iterasi (PCA 2D)", "Komponen PCA 1", "Komponen PCA 2", height=HEIGHT_MAIN)
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0), margin=dict(l=60, r=20, t=60, b=100))
    return fig


def plot_membership_function(grid: np.ndarray, U_grid: np.ndarray, feature_label: str,
                             cluster_names: Sequence[str] | None = None,
                             x_marker: float | None = None) -> go.Figure:
    """Fungsi keanggotaan FCM terhadap satu fitur (hitungan eksak, bukan rata-rata data).

    ``grid`` = nilai fitur (n_titik,), ``U_grid`` = derajat keanggotaan (n_titik, c) yang dihitung
    ``fcm.membership`` ketika fitur lain ditahan tetap. Mirip grafik fungsi keanggotaan fuzzy
    pada logika fuzzy klasik, tetapi bentuknya diturunkan dari pusat cluster.
    """
    U_grid = np.asarray(U_grid, dtype=float)
    names = list(cluster_names) if cluster_names is not None else _default_names(U_grid.shape[1])
    fig = go.Figure()
    for j in range(U_grid.shape[1]):
        fig.add_trace(go.Scatter(x=grid, y=U_grid[:, j], mode="lines", name=names[j],
                                 line=dict(color=cluster_color(j), width=3),
                                 hovertemplate=f"{feature_label} = %{{x:.3f}}<br>u = %{{y:.3f}}<extra>{names[j]}</extra>"))
    if x_marker is not None:
        fig.add_vline(x=float(x_marker), line=dict(color=BASELINE_COLOR, dash="dot"),
                      annotation_text="film rata-rata", annotation_position="top")
    _layout(fig, f"Fungsi keanggotaan cluster terhadap {feature_label}",
            f"{feature_label} (ternormalisasi; fitur lain tetap)", "Derajat keanggotaan u")
    fig.update_yaxes(range=[0, 1.02])
    return fig


def plot_contoh_fcm(X: np.ndarray, nama: Sequence[str], V_list: Sequence[np.ndarray],
                    labels: Sequence[str], U: np.ndarray) -> go.Figure:
    """Peta kecil contoh perhitungan: 5 film, pusat cluster awal → akhir, label film.

    Dipakai pada tab Contoh Perhitungan (Informasi). Warna titik = cluster dominan menurut ``U``.
    """
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    c = U.shape[1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers+text", text=list(nama), textposition="top center",
                             marker=dict(size=16, color=[cluster_color(int(j)) for j in U.argmax(axis=1)],
                                         line=dict(color=TEXT_COLOR, width=1.5)),
                             name="Film", hovertemplate="Film %{text}<br>(%{x:.2f}, %{y:.2f})<extra></extra>"))
    for j in range(c):
        xs = [v[j, 0] for v in V_list]
        ys = [v[j, 1] for v in V_list]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"Pusat C{j + 1}", customdata=labels,
                                 line=dict(color=cluster_color(j), dash="dot"),
                                 marker=dict(symbol="x", size=13, color=cluster_color(j),
                                             line=dict(width=3, color=cluster_color(j))),
                                 hovertemplate="%{customdata}<br>(%{x:.3f}, %{y:.3f})<extra>C" + str(j + 1) + "</extra>"))
    _layout(fig, "Contoh: film dan perpindahan pusat cluster", "Rating (0–1)", "Popularitas (0–1)")
    fig.update_xaxes(range=[0, 1.05])
    fig.update_yaxes(range=[0, 1.05])
    return fig


def alur_algoritma_dot() -> str:
    """Flowchart algoritma FCM + KNN fuzzy (untuk halaman Informasi dan BAB III) dalam format DOT."""
    return _tema_dot("""
digraph G {
  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.3;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10,
        color="__BORDER__", fillcolor="__SURFACE__", fontcolor="__TEXT__"];
  edge [color="__MUTED__", fontcolor="__MUTED__", fontname="Helvetica", fontsize=9];
  mulai [label="Mulai", shape=oval];
  x [label="Matriks fitur X (n × f)\\nparameter c, m, ε, maxiter"];
  init [label="Inisialisasi pusat / U awal\\n(seed 42)"];
  v [label="Hitung pusat cluster\\nv_j = Σ u^m x / Σ u^m"];
  u [label="Perbarui keanggotaan\\nu_ij = 1 / Σ (d_ij/d_ik)^(2/(m−1))"];
  j [label="Hitung J_m dan Δ = max|U_baru − U_lama|"];
  cek [label="Δ < ε atau iterasi = maxiter?", shape=diamond, fillcolor="__BORDER__"];
  out [label="Keluaran: V, U, J_history, FPC", fillcolor="__PRIMARY__"];
  q [label="Film acuan q → vektor keanggotaan u_q"];
  d [label="Jarak KNN fuzzy\\nd = α‖x_q − x_i‖ + (1−α)‖u_q − u_i‖", fillcolor="__PRIMARY__"];
  top [label="Urutkan jarak, ambil k terkecil\\nskor = 1 / (1 + d)"];
  hasil [label="Tabel rekomendasi + interpretasi"];
  selesai [label="Selesai", shape=oval];
  mulai -> x -> init -> v -> u -> j -> cek;
  cek -> v [label="tidak"];
  cek -> out [label="ya"];
  out -> q -> d -> top -> hasil -> selesai;
}
""")
