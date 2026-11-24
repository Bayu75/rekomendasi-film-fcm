"""CSS tema Cinematic Dark FuzzyFlix (latar #0B0D12, kartu #151821, aksen merah #E50914).

Anggota 4: GUI & Visualisasi. Satu blok CSS dipasang di setiap halaman lewat ``ui.page_setup``.
Streamlit tidak menyediakan API untuk menata sidebar dan navigasi, jadi selektor di sini
memakai atribut ``data-testid`` yang stabil; bila versi Streamlit berubah, cek bagian ini dahulu.
"""

from __future__ import annotations

BG = "#0B0D12"
SIDEBAR = "#10131A"
KARTU = "#151821"
BORDER = "#262B38"
MERAH = "#E50914"
TEKS = "#F5F5F7"
REDUP = "#A0A6B1"

# Hiasan dasar sidebar: gelombang dan gulungan film merah samar (SVG sebagai data URI).
_DEKOR_SIDEBAR = (
    "data:image/svg+xml;utf8,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 260'%3E"
    "%3Cg fill='none' stroke='%23E50914' stroke-linecap='round'%3E"
    "%3Cpath d='M-10 190 C40 150 80 230 140 190 S250 150 320 190' stroke-opacity='.30' stroke-width='1.4'/%3E"
    "%3Cpath d='M-10 206 C40 166 80 246 140 206 S250 166 320 206' stroke-opacity='.22' stroke-width='1.4'/%3E"
    "%3Cpath d='M-10 222 C40 182 80 262 140 222 S250 182 320 222' stroke-opacity='.15' stroke-width='1.4'/%3E"
    "%3Cpath d='M-10 238 C40 198 80 278 140 238 S250 198 320 238' stroke-opacity='.10' stroke-width='1.4'/%3E"
    "%3Ccircle cx='230' cy='205' r='44' stroke-opacity='.20' stroke-width='2'/%3E"
    "%3Ccircle cx='230' cy='205' r='9' stroke-opacity='.25' stroke-width='2'/%3E"
    "%3Cpath d='M230 170v22M230 218v22M195 205h22M243 205h22' stroke-opacity='.18' stroke-width='3'/%3E"
    "%3C/g%3E%3C/svg%3E"
)

CSS = f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap");

/* ---------- Dasar ---------- */
.ff-i {{ display: inline-block; flex: none; vertical-align: middle; -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
  -webkit-mask-position: center; mask-position: center; -webkit-mask-size: contain; mask-size: contain; }}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"], [data-testid="stHeader"], button, input, textarea, h1, h2, h3, h4, h5, [data-testid="stMarkdownContainer"] * {{
  font-family: "Inter", "Segoe UI", system-ui, sans-serif !important; }}
[data-testid="stApp"], [data-testid="stAppViewContainer"] {{ background: {BG}; }}
[data-testid="stHeader"] {{ background: rgba(11,13,18,.94) !important; backdrop-filter: blur(6px); height: 3.2rem; }}
[data-testid="stMainBlockContainer"], .block-container {{ max-width: 1240px; padding: 3.6rem 2.2rem 4rem; }}
h2, h3 {{ font-weight: 700; color: {TEKS} !important; letter-spacing: -0.01em; }}
[data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {{ border-left: 3px solid {MERAH}; padding-left: 12px; line-height: 1.25; margin-top: 0.4rem; }}
[data-testid="stHeading"] h2 {{ font-size: 1.55rem; }} [data-testid="stHeading"] h3 {{ font-size: 1.2rem; }}
[data-testid="stCaptionContainer"] {{ color: {REDUP}; }}
hr {{ border-color: {BORDER} !important; }}
a {{ color: #FF5A63; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
  background-color: {SIDEBAR};
  background-image: url("{_DEKOR_SIDEBAR}");
  background-repeat: no-repeat; background-position: left bottom; background-size: 100% auto;
  border-right: 1px solid {BORDER}; min-width: 280px; max-width: 280px; }}
[data-testid="stSidebar"] > div:first-child {{ background: transparent; }}
[data-testid="stSidebarContent"] {{ background: transparent; padding: 0 !important; }}
[data-testid="stSidebarHeader"] {{ height: 0; min-height: 0; padding: 0; }}
[data-testid="stSidebarUserContent"] {{ padding: 1.3rem 1rem 1rem !important; margin: 0 !important; width: 100% !important; }}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {{ gap: 0.35rem; min-height: calc(100vh - 3rem); }}
.ff-logo {{ display: flex; align-items: center; gap: 10px; padding: 2px 6px 18px; }}
.ff-logo-teks {{ font-size: 1.72rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.03em; }}
.ff-logo-teks span {{ color: {MERAH}; }}
.ff-logo-ikon {{ filter: drop-shadow(0 0 8px rgba(229,9,20,.55)); display: flex; }}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{
  border-radius: 10px; padding: 0.62rem 0.85rem; color: #E8EAF0; font-weight: 500; gap: 0.7rem;
  transition: background .15s; }}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {{ background: rgba(255,255,255,0.06); }}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] p {{ font-size: 0.97rem; font-weight: 500; color: inherit; }}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] {{ font-size: 1.3rem; color: inherit; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] {{ gap: 0; width: 100%; }}
[data-testid="stSidebar"] [class*="st-key-nav_"] [data-testid="stPageLink"], [data-testid="stSidebar"] [class*="st-key-nav_"] a {{ width: 100%; }}
[data-testid="stSidebar"] [data-testid="stLayoutWrapper"]:has([class*="st-key-nav_"]), [data-testid="stSidebar"] [data-testid="stElementContainer"]:has([class*="st-key-nav_"]) {{ width: 100% !important; }}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] p {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

.ff-status {{ margin-top: 1.1rem; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 14px; background: rgba(21,24,33,.72); }}
.ff-status-judul {{ font-weight: 650; font-size: 0.92rem; color: #FFFFFF; margin-bottom: 8px; }}
.ff-status-baris {{ display: flex; align-items: center; gap: 9px; font-size: 0.8rem; white-space: nowrap; color: #D5D8E0; padding: 3px 0; }}
.ff-titik {{ width: 8px; height: 8px; border-radius: 50%; background: #6B7280; flex: none; }}
.ff-titik.ok {{ background: #22C55E; box-shadow: 0 0 8px rgba(34,197,94,.9); }}
.ff-titik.warn {{ background: #F59E0B; box-shadow: 0 0 8px rgba(245,158,11,.8); }}
.ff-cek {{ display: flex; flex: none; }}
.ff-tagline {{ font-style: italic; color: {REDUP}; font-size: 0.86rem; padding: 0 8px; }}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.ff-tagline) {{ margin-top: auto; padding-bottom: 0.4rem; }}

/* ---------- Bilah atas kanan ---------- */
.ff-topbar {{ position: fixed; top: 12px; right: 70px; z-index: 1000002; display: flex; align-items: center; gap: 12px; }}
.ff-bulan {{ display: flex; }}
.ff-chip {{ display: inline-flex; align-items: center; gap: 8px; background: {KARTU}; border: 1px solid {BORDER};
  border-radius: 999px; padding: 5px 14px 5px 10px; font-size: 0.86rem; color: #E8EAF0; }}

/* ---------- Header halaman ---------- */
.ff-header {{ position: relative; display: flex; justify-content: space-between; align-items: center; min-height: 140px;
  margin: 0 0 1.1rem; border: 1px solid {BORDER}; border-radius: 16px; overflow: hidden;
  background: linear-gradient(100deg, {KARTU} 0%, #14121A 55%, #2A0A10 100%); }}
.ff-header-teks {{ padding: 20px 28px; position: relative; z-index: 2; }}
.ff-judul-baris {{ display: flex; align-items: center; gap: 14px; }}
.ff-judul-ikon {{ display: flex; filter: drop-shadow(0 0 10px rgba(229,9,20,.6)); }}
h1.ff-judul {{ margin: 0; padding: 0; font-size: 2.55rem; font-weight: 800; letter-spacing: -0.035em; color: #FFFFFF !important; line-height: 1.1; }}
h1.ff-judul span {{ color: {MERAH}; }}
.ff-subjudul {{ margin: 8px 0 0 48px; font-style: italic; color: {REDUP}; font-size: 1.02rem; }}
.ff-hero {{ position: absolute; right: 0; top: 0; bottom: 0; width: 50%; -webkit-mask-image: linear-gradient(90deg, transparent 0, #000 30%);
  mask-image: linear-gradient(90deg, transparent 0, #000 30%); }}
.ff-hero-svg {{ width: 100%; height: 100%; background-size: auto 100%; background-position: right center; background-repeat: no-repeat; }}

/* ---------- Kartu ---------- */
[class*="st-key-kartu"], div[data-testid="stExpander"], .ff-kartu {{
  background: {KARTU}; border: 1px solid {BORDER} !important; border-radius: 12px; }}
[class*="st-key-kartu"] {{ padding: 1rem 1.15rem 0.7rem; box-shadow: 0 8px 30px rgba(0,0,0,.28); }}
div[data-testid="stExpander"] details {{ border: none; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {BORDER} !important; border-radius: 12px; }}
.ff-kartu {{ padding: 16px 18px; height: 100%; }}
.ff-kartu h4 {{ margin: 0 0 6px 0; color: #FFFFFF; }}
.ff-kt {{ display: flex; align-items: center; gap: 9px; font-weight: 700; font-size: 1.04rem; color: #FFFFFF; margin: 2px 0 12px; }}
.ff-kt .ff-ikon {{ flex: none; }}
[data-testid="stElementContainer"]:has(> [data-testid="stHtml"] > style) {{ display: none; }}

/* ---------- Callout Ringkasan Hasil ---------- */
.ff-callout {{ display: flex; gap: 16px; align-items: flex-start; border-radius: 12px; padding: 16px 20px;
  border: 1px solid rgba(229,9,20,.7); box-shadow: 0 0 28px rgba(229,9,20,.18);
  background: linear-gradient(100deg, rgba(229,9,20,.30) 0%, rgba(120,6,12,.22) 55%, rgba(21,24,33,.9) 100%); }}
.ff-callout-ikon {{ display: flex; margin-top: 2px; filter: drop-shadow(0 0 8px rgba(251,191,36,.5)); }}
.ff-callout-judul {{ font-weight: 700; color: #FFFFFF; margin-bottom: 4px; }}
.ff-callout-isi {{ color: #E6E8EE; line-height: 1.55; font-size: 0.97rem; }}

/* ---------- Film Pilihan ---------- */
.ff-kartu-lebar {{ background: {KARTU}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 20px 18px; box-shadow: 0 8px 30px rgba(0,0,0,.28); }}
.ff-pilihan {{ display: flex; gap: 0; align-items: stretch; }}
.ff-pilihan-film {{ display: flex; gap: 18px; flex: 1 1 auto; min-width: 0; }}
.ff-pilihan-info {{ min-width: 0; }}
.ff-pilihan-judul {{ font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; color: #FFFFFF; margin-bottom: 8px; }}
.ff-chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
.ff-chip-info, .ff-chip-genre {{ display: inline-flex; align-items: center; gap: 5px; border-radius: 8px; padding: 3px 10px; font-size: 0.8rem;
  background: #1D212C; border: 1px solid {BORDER}; color: #E8EAF0; }}
.ff-chip-genre {{ background: rgba(229,9,20,.12); border-color: rgba(229,9,20,.35); color: #FFD5D8; }}
.ff-sinopsis {{ margin: 0; color: #C5C9D3; font-size: 0.92rem; line-height: 1.55; display: -webkit-box; -webkit-line-clamp: 5; -webkit-box-orient: vertical; overflow: hidden; }}
.ff-redup {{ color: {REDUP}; font-style: italic; }}
.ff-garis-v {{ width: 1px; background: {BORDER}; margin: 0 24px; flex: none; }}
.ff-pilihan-prm {{ flex: 0 0 250px; }}
.ff-prm-judul {{ font-weight: 700; color: #FFFFFF; margin-bottom: 10px; }}
.ff-prm {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #202533; font-size: 0.9rem; color: {REDUP}; }}
.ff-prm:last-child {{ border-bottom: none; }}
.ff-prm b {{ color: #FFFFFF; font-size: 1rem; }}

/* ---------- Poster ---------- */
.ff-poster {{ position: relative; border-radius: 8px; overflow: hidden; flex: none; display: flex; align-items: center; justify-content: center;
  border: 1px solid rgba(255,255,255,.08); box-shadow: 0 6px 18px rgba(0,0,0,.45); }}
.ff-poster span {{ color: rgba(255,255,255,.85); font-weight: 800; letter-spacing: .04em; }}
.ff-poster-img {{ position: absolute; inset: 0; background-size: cover; background-position: center; background-repeat: no-repeat; }}

/* ---------- Kartu metrik ---------- */
.ff-baris-metrik {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 14px 0; }}
.ff-metrik {{ background: {KARTU}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px 0; overflow: hidden; }}
.ff-metrik-atas {{ display: flex; gap: 12px; align-items: center; }}
.ff-metrik-ikon {{ width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; flex: none;
  background: color-mix(in srgb, var(--w) 16%, transparent); border: 1px solid color-mix(in srgb, var(--w) 40%, transparent); }}
.ff-metrik-nilai {{ font-size: 1.75rem; font-weight: 800; color: #FFFFFF; line-height: 1.1; letter-spacing: -0.02em; }}
.ff-metrik-label {{ font-size: 0.8rem; color: {REDUP}; margin-top: 2px; }}
.ff-spark {{ display: block; width: calc(100% + 32px); margin: 6px -16px 0; height: 46px; background-size: 100% 100%; background-repeat: no-repeat; }}

/* ---------- Tabel rekomendasi ---------- */
.ff-tabel {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
.ff-tabel th {{ text-align: left; font-weight: 600; color: {REDUP}; font-size: 0.8rem; padding: 6px 10px 10px; border-bottom: 1px solid {BORDER}; }}
.ff-tabel td {{ padding: 8px 10px; border-bottom: 1px solid #1E2330; color: #E8EAF0; vertical-align: middle; }}
.ff-tabel tr:last-child td {{ border-bottom: none; }}
.ff-tabel .c {{ text-align: center; }}
.ff-no {{ color: {REDUP}; }}
.ff-judul-sel {{ display: flex; align-items: center; gap: 12px; font-weight: 600; color: #FFFFFF; }}
.ff-judul-sel .ff-poster {{ border-radius: 5px; }}
.ff-genre-sel {{ color: {REDUP}; }}
.ff-skor {{ display: flex; align-items: center; gap: 12px; min-width: 150px; }}
.ff-skor b {{ color: #FFFFFF; font-size: 0.95rem; min-width: 44px; }}
.ff-bar {{ flex: 1; height: 6px; background: #262B38; border-radius: 99px; overflow: hidden; }}
.ff-bar i {{ display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, #B20710, {MERAH}); box-shadow: 0 0 10px rgba(229,9,20,.6); }}

/* ---------- Beranda ---------- */
.ff-langkah {{ background: {KARTU}; border: 1px solid {BORDER}; border-radius: 12px; padding: 18px; height: 100%; position: relative; }}
.ff-langkah-ikon {{ width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
  background: rgba(229,9,20,.12); border: 1px solid rgba(229,9,20,.4); margin-bottom: 10px; }}
.ff-langkah-no {{ position: absolute; top: 14px; right: 18px; font-size: 2rem; font-weight: 800; color: rgba(255,255,255,.08); }}
.ff-langkah h4 {{ margin: 0 0 6px; color: #FFFFFF; font-size: 1.05rem; }}
.ff-langkah p {{ margin: 0; color: #C5C9D3; font-size: 0.92rem; line-height: 1.5; }}

/* ---------- Widget bawaan ---------- */
button[kind="primary"] {{ background: {MERAH}; border: 1px solid {MERAH}; color: #FFFFFF; font-weight: 650; border-radius: 10px;
  box-shadow: 0 4px 18px rgba(229,9,20,.35); }}
button[kind="primary"]:hover {{ background: #C40812 !important; border-color: #C40812 !important; }}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {{ background: transparent; border: 1px solid {MERAH}; color: #FF6B73; border-radius: 10px; font-weight: 600; }}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {{ background: rgba(229,9,20,.12); border-color: {MERAH}; color: #FFFFFF; }}
[data-testid="stDownloadButton"] button {{ border-radius: 10px; }}
[data-testid="stMetric"] {{ background: {KARTU}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px; }}
[data-testid="stMetricValue"] {{ font-size: 1.75rem; font-weight: 800; color: #FFFFFF; }}
[data-testid="stMetricLabel"] {{ color: {REDUP}; }}
div[data-testid="stAlert"] {{ border-radius: 12px; }}
[data-baseweb="tab-list"] {{ gap: 6px; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: #FF6B73; }}
[data-baseweb="tab-highlight"] {{ background-color: {MERAH} !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}
[data-testid="stSlider"] [role="slider"] {{ background-color: {MERAH}; }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{ background: #10131A; border-color: {BORDER}; border-radius: 10px; }}
.ff-demo {{ border: 1px dashed rgba(245,158,11,.55); background: rgba(245,158,11,.07); border-radius: 12px; padding: 10px 14px; margin-bottom: 12px; color: #F3E3C3; font-size: 0.9rem; }}
.ff-demo b {{ color: #FBBF24; }}
.ff-demo ul {{ margin: 6px 0 4px 1.1rem; padding: 0; }}
</style>
"""


def css_menu_aktif(aktif: str | None) -> str:
    """CSS pil merah untuk item menu halaman aktif (Streamlit tidak menandai halaman aktif pada ``page_link``)."""
    if not aktif or not aktif.isalpha():
        return ""
    return (f"<style>.st-key-nav_{aktif} a[data-testid=\"stPageLink-NavLink\"] {{ background: {MERAH} !important; color: #FFFFFF !important;"
            f" box-shadow: 0 6px 22px rgba(229,9,20,.38); }} .st-key-nav_{aktif} a p {{ font-weight: 650 !important; }}</style>")
