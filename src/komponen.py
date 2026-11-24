"""Komponen tampilan HTML/SVG tanpa dependensi tambahan (tema Cinematic Dark).

Anggota 4: GUI & Visualisasi. Semua fungsi mengembalikan string HTML yang siap
dipasang lewat ``st.html``. Teks dari data (judul film, genre, sinopsis) selalu
di-escape; tidak ada gambar berhak cipta, ikon dan ilustrasi digambar dengan SVG.
``st.html`` membersihkan tag ``<svg>`` dari HTML, jadi semua grafik SVG dipasang sebagai
gambar data-URI lewat CSS (``mask-image`` untuk ikon, ``background-image`` untuk sparkline
dan ilustrasi). Poster film memakai ``https://image.tmdb.org/t/p/w185{poster_path}`` bila ada;
bila kosong atau tidak bisa dimuat (offline, mode demo), tampil poster pengganti
berisi inisial judul, jadi tidak ada panggilan jaringan di sisi Python.
"""

from __future__ import annotations

import hashlib
import html
import math
import re
from typing import Sequence
from urllib.parse import quote

TMDB_URL = "https://image.tmdb.org/t/p/w185"
MERAH = "#E50914"
BIRU = "#3B82F6"
HIJAU = "#22C55E"
AMBER = "#F59E0B"

_POLA_POSTER = re.compile(r"^/[A-Za-z0-9_\-.]+$")


def esc(teks) -> str:
    """Escape teks dinamis untuk HTML (None menjadi string kosong)."""
    return "" if teks is None else html.escape(str(teks), quote=True)


def md_mini(teks) -> str:
    """Escape lalu ubah ``**tebal**`` jadi <b>, `kode` jadi <code>, dan baris baru jadi <br>."""
    aman = esc(teks)
    aman = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", aman)
    aman = re.sub(r"`(.+?)`", r"<code>\1</code>", aman)
    return aman.replace("\n", "<br>")


# ---------------------------------------------------------------------------
# Ikon (SVG garis, viewBox 24)
# ---------------------------------------------------------------------------

_IKON = {
    "home": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v10h4.5v-6h4v6h4.5V10"/>',
    "database": '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>'
                '<path d="M4 11.5v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    "gear": '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="7"/>'
            '<path d="M12 2.5v2.5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8"/>',
    "star": '<path d="M12 2.8l2.9 6 6.6.9-4.8 4.6 1.2 6.6L12 17.7l-5.9 3.2 1.2-6.6L2.5 9.7l6.6-.9z"/>',
    "chart": '<path d="M4 20V11M10 20V4M16 20v-7M22 20H2"/>',
    "info": '<circle cx="12" cy="12" r="9.5"/><path d="M12 16.5v-5M12 7.8h.01"/>',
    "clapper": '<rect x="3" y="9.5" width="18" height="11" rx="2"/><path d="M3.5 9.5 5 4.2l4.3 1.1L7.7 9.5M9.3 5.3l4.3 1.1-1.6 3.1M13.6 6.4l4.3 1.1-1.6 2M17.9 7.5l2.6.7-.7 1.3"/>',
    "moon": '<path d="M20.5 13.2A8.5 8.5 0 1 1 10.8 3.5a6.6 6.6 0 0 0 9.7 9.7z"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-6.2 8-6.2s8 2.2 8 6.2"/>',
    "bulb": '<path d="M9 18.5h6M10 21.5h4"/><path d="M12 2.8a6.2 6.2 0 0 0-4 10.9c.9.8 1.2 1.6 1.2 2.8h5.6c0-1.2.3-2 1.2-2.8A6.2 6.2 0 0 0 12 2.8z"/>',
    "network": '<circle cx="12" cy="5" r="2.4"/><circle cx="5" cy="18" r="2.4"/><circle cx="19" cy="18" r="2.4"/>'
               '<path d="M12 7.4v4.2M12 11.6 6.4 16M12 11.6l5.6 4.4"/><circle cx="12" cy="11.6" r="1.4"/>',
    "bars": '<path d="M5 20v-7M12 20V4M19 20v-10"/>',
    "target": '<circle cx="12" cy="12" r="9.5"/><circle cx="12" cy="12" r="5.5"/><circle cx="12" cy="12" r="1.6"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5.2l3.2 2"/>',
    "calendar": '<rect x="3.5" y="5" width="17" height="15.5" rx="2"/><path d="M3.5 10h17M8 3v4M16 3v4"/>',
    "film": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01"/>',
    "scatter": '<circle cx="7" cy="16" r="1.7"/><circle cx="12" cy="9" r="1.7"/><circle cx="17" cy="14" r="1.7"/><circle cx="18" cy="6" r="1.7"/><path d="M3 3v18h18"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-9"/><path d="M15 6h6v6"/>',
    "check": '<path d="M4.5 12.5l5 5 10-11"/>',
}


def _uri(svg: str) -> str:
    """SVG -> data URI aman untuk dipakai di ``url('...')`` pada atribut style."""
    return "data:image/svg+xml;utf8," + quote(svg, safe="")


def ikon(nama: str, warna: str = "currentColor", ukuran: int = 20, tebal: float = 1.8, isi: str = "none") -> str:
    """Ikon bergaya garis sebagai elemen ``span`` bertopeng SVG (warna dari ``warna``).

    ``nama`` yang tidak dikenal memakai ikon ``info``. ``isi != "none"`` membuat ikon solid.
    """
    badan = _IKON.get(nama, _IKON["info"])
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{"black" if isi != "none" else "none"}" '
           f'stroke="black" stroke-width="{tebal}" stroke-linecap="round" stroke-linejoin="round">{badan}</svg>')
    u = _uri(svg)
    return (f'<span class="ff-i" role="img" aria-label="{esc(nama)}" style="width:{ukuran}px;height:{ukuran}px;background-color:{warna};'
            f"-webkit-mask-image:url('{u}');mask-image:url('{u}')\"></span>")


# ---------------------------------------------------------------------------
# Poster
# ---------------------------------------------------------------------------

def url_poster(poster_path) -> str | None:
    """URL TMDB w185 bila ``poster_path`` valid (``/abc.jpg``), jika tidak None."""
    if poster_path is None:
        return None
    teks = str(poster_path).strip()
    if teks.lower() in ("", "nan", "none") or not _POLA_POSTER.match(teks):
        return None
    return TMDB_URL + teks


def inisial(judul) -> str:
    """Maksimal dua huruf awal kata judul, huruf besar; ``?`` bila kosong."""
    kata = [k for k in re.findall(r"[^\W_]+", str(judul or ""), flags=re.UNICODE) if k]
    return "".join(k[0] for k in kata[:2]).upper() or "?"


def _gradien_poster(judul) -> str:
    h = int(hashlib.md5(str(judul).encode("utf-8")).hexdigest()[:6], 16)
    nada = (350, 8, 220, 262, 330, 200)[h % 6]
    return (f"linear-gradient(155deg, hsl({nada} 55% 24%) 0%, hsl({(nada + 30) % 360} 45% 11%) 100%)")


def poster(judul, poster_path=None, lebar: int = 120, tinggi: int | None = None, kelas: str = "") -> str:
    """Poster: gambar TMDB (lapisan background) di atas poster pengganti (gradien gelap + inisial judul)."""
    tinggi = tinggi or int(lebar * 1.5)
    url = url_poster(poster_path)
    # gambar sebagai background lapisan atas: bila gagal dimuat (offline) lapisan transparan dan inisial tetap terlihat
    gambar = (f'<div class="ff-poster-img" style="background-image:url(\'{esc(url)}\')"></div>' if url else "")
    ukuran_huruf = max(11, int(lebar * 0.3))
    return (f'<div class="ff-poster {esc(kelas)}" style="width:{lebar}px;height:{tinggi}px;background:{_gradien_poster(judul)};">'
            f'<span style="font-size:{ukuran_huruf}px">{esc(inisial(judul))}</span>{gambar}</div>')


# ---------------------------------------------------------------------------
# Grafik garis mini (sparkline) dan ilustrasi
# ---------------------------------------------------------------------------

def _jalur_halus(titik: Sequence[tuple[float, float]]) -> str:
    if len(titik) < 2:
        return ""
    d = f"M{titik[0][0]:.1f},{titik[0][1]:.1f}"
    for (x0, y0), (x1, y1) in zip(titik, titik[1:]):
        xm = (x0 + x1) / 2
        d += f" C{xm:.1f},{y0:.1f} {xm:.1f},{y1:.1f} {x1:.1f},{y1:.1f}"
    return d


_GELOMBANG = (0.35, 0.55, 0.42, 0.7, 0.5, 0.82, 0.62, 0.9)


def sparkline(nilai: Sequence[float] | None, warna: str, lebar: int = 220, tinggi: int = 46, id_: str = "s") -> str:
    """Garis tipis berwarna dengan isi gradien. Tanpa data memakai gelombang dekoratif."""
    seri = [float(v) for v in (nilai or []) if v is not None and math.isfinite(float(v))]
    if len(seri) < 2:
        seri = list(_GELOMBANG)
    lo, hi = min(seri), max(seri)
    rentang = (hi - lo) or 1.0
    pad = 6
    titik = [(pad + i * (lebar - 2 * pad) / (len(seri) - 1),
              pad + (1 - (v - lo) / rentang) * (tinggi - 2 * pad)) for i, v in enumerate(seri)]
    garis = _jalur_halus(titik)
    isi = f"{garis} L{titik[-1][0]:.1f},{tinggi} L{titik[0][0]:.1f},{tinggi} Z"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {lebar} {tinggi}" preserveAspectRatio="none">'
           f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
           f'<stop offset="0" stop-color="{warna}" stop-opacity="0.38"/><stop offset="1" stop-color="{warna}" stop-opacity="0"/>'
           f'</linearGradient></defs><path d="{isi}" fill="url(#g)"/>'
           f'<path d="{garis}" fill="none" stroke="{warna}" stroke-width="2" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>')
    return f"<div class=\"ff-spark\" role=\"img\" aria-label=\"grafik garis mini\" style=\"background-image:url('{_uri(svg)}')\"></div>"


_KAMERA_SVG = '''<svg viewBox="0 0 420 150" preserveAspectRatio="xMaxYMid slice" xmlns="http://www.w3.org/2000/svg">
<defs>
<radialGradient id="hg1" cx="62%" cy="45%" r="65%"><stop offset="0" stop-color="#B3121C" stop-opacity=".75"/><stop offset=".55" stop-color="#5A0A10" stop-opacity=".55"/><stop offset="1" stop-color="#0B0D12" stop-opacity="0"/></radialGradient>
<filter id="kabut" x="-20%" y="-30%" width="140%" height="160%"><feGaussianBlur stdDeviation="9"/></filter>
</defs>
<rect width="420" height="150" fill="url(#hg1)"/>
<g filter="url(#kabut)" opacity=".75"><ellipse cx="120" cy="108" rx="80" ry="22" fill="#7A0D14"/><ellipse cx="300" cy="125" rx="96" ry="20" fill="#4C070C"/><ellipse cx="230" cy="40" rx="70" ry="16" fill="#8E1018"/></g>
<g fill="#07080C" opacity=".96">
<circle cx="235" cy="36" r="27"/><circle cx="288" cy="40" r="22"/>
<rect x="200" y="56" width="124" height="58" rx="9"/>
<path d="M200 70 154 54v46l46-14z"/><rect x="144" y="58" width="14" height="36" rx="3"/>
<path d="M232 114l-18 30M292 114l18 30M262 114v30" stroke="#07080C" stroke-width="5" stroke-linecap="round"/>
<rect x="324" y="72" width="22" height="26" rx="4"/>
</g>
<g fill="none" stroke="#E50914" stroke-opacity=".55" stroke-width="1.4"><circle cx="235" cy="36" r="19"/><circle cx="288" cy="40" r="15"/><path d="M235 17v38M216 36h38M288 25v30M273 40h30" stroke-opacity=".28"/></g>
<circle cx="150" cy="76" r="5" fill="#E50914" opacity=".85"/>
</svg>'''


def ilustrasi_kamera() -> str:
    """Siluet kamera film di atas asap merah gelap (SVG buatan sendiri, tanpa gambar eksternal)."""
    return f"<div class=\"ff-hero-svg\" role=\"img\" aria-label=\"ilustrasi kamera film\" style=\"background-image:url('{_uri(_KAMERA_SVG)}')\"></div>"


# ---------------------------------------------------------------------------
# Komponen halaman
# ---------------------------------------------------------------------------

def judul_berwarna(judul: str) -> tuple[str, str]:
    """Pecah judul: bagian putih dan bagian merah. ``FuzzyFlix`` -> (``Fuzzy``, ``Flix``)."""
    kata = judul.split()
    if len(kata) >= 2:
        return kata[0], " ".join(kata[1:])
    m = re.match(r"^(.+?[a-z])([A-Z].*)$", judul)
    return (m.group(1), m.group(2)) if m else (judul, "")


def header_halaman(judul: str, subjudul: str, nama_ikon: str = "clapper") -> str:
    """Header: ikon klaker merah, judul dua warna, subjudul miring, hero kamera di kanan."""
    putih, merah = judul_berwarna(judul)
    return (f'<div class="ff-header"><div class="ff-header-teks">'
            f'<div class="ff-judul-baris"><span class="ff-judul-ikon">{ikon(nama_ikon, MERAH, 34, 1.7)}</span>'
            f'<h1 class="ff-judul">{esc(putih)}{" " if " " in judul else ""}<span>{esc(merah)}</span></h1></div>'
            f'<p class="ff-subjudul">{esc(subjudul)}</p></div>'
            f'<div class="ff-hero">{ilustrasi_kamera()}</div></div>')


def logo() -> str:
    return (f'<div class="ff-logo"><span class="ff-logo-ikon">{ikon("clapper", MERAH, 30, 1.7)}</span>'
            f'<span class="ff-logo-teks">Fuzzy<span>Flix</span></span></div>')


def kartu_status(baris: Sequence[tuple[str, str]]) -> str:
    """Kartu "Status Sistem". ``baris`` berisi (teks, kondisi) dengan kondisi ok/peringatan/info."""
    isi = ""
    for teks, kondisi in baris:
        tanda = {"ok": '<span class="ff-titik ok"></span>', "cek": f'<span class="ff-cek">{ikon("check", HIJAU, 14, 2.6)}</span>',
                 "peringatan": '<span class="ff-titik warn"></span>'}.get(kondisi, '<span class="ff-titik"></span>')
        isi += f'<div class="ff-status-baris">{tanda}<span>{esc(teks)}</span></div>'
    return f'<div class="ff-status"><div class="ff-status-judul">Status Sistem</div>{isi}</div>'


def chip_pengguna(nama: str = "FuzzyFlix") -> str:
    return (f'<div class="ff-topbar"><span class="ff-bulan">{ikon("moon", "#C9CDD6", 18, 1.7)}</span>'
            f'<span class="ff-chip">{ikon("user", MERAH, 16, 2)}<span>{esc(nama)}</span></span></div>')


def judul_kartu(teks: str, nama_ikon: str, warna: str = MERAH) -> str:
    return f'<div class="ff-kt">{ikon(nama_ikon, warna, 18, 1.9)}<span>{esc(teks)}</span></div>'


def callout_ringkasan(teks_md: str) -> str:
    """Kartu "Ringkasan Hasil". ``teks_md`` berupa teks biasa; ``**tebal**`` diubah menjadi <b>."""
    aman = md_mini(teks_md)
    return (f'<div class="ff-callout"><span class="ff-callout-ikon">{ikon("bulb", "#FBBF24", 26, 1.8)}</span>'
            f'<div><div class="ff-callout-judul">Ringkasan Hasil</div><div class="ff-callout-isi">{aman}</div></div></div>')


def kartu_film_pilihan(judul, tahun, rating, durasi, genre: Sequence[str], sinopsis, poster_path,
                       parameter: Sequence[tuple[str, str]]) -> str:
    """Kartu "Film Pilihan": poster, judul, chip info, sinopsis | garis | Parameter Rekomendasi."""
    chips = [f'<span class="ff-chip-info">{ikon("star", AMBER, 13, 1.6, AMBER)}{esc(rating)}/10</span>']
    if tahun:
        chips.append(f'<span class="ff-chip-info">{ikon("calendar", "#C9CDD6", 13, 1.8)}{esc(tahun)}</span>')
    if durasi:
        chips.append(f'<span class="ff-chip-info">{ikon("clock", "#C9CDD6", 13, 1.8)}{esc(durasi)}</span>')
    chips += [f'<span class="ff-chip-genre">{esc(g)}</span>' for g in genre]
    sin = (f'<p class="ff-sinopsis">{esc(sinopsis)}</p>' if sinopsis and str(sinopsis).strip().lower() not in ("nan", "none")
           else '<p class="ff-sinopsis ff-redup">Sinopsis tidak tersedia pada dataset ini.</p>')
    prm = "".join(f'<div class="ff-prm"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in parameter)
    return (f'<div class="ff-kartu-lebar"><div class="ff-kt">{ikon("star", MERAH, 18, 1.9, MERAH)}<span>Film Pilihan</span></div>'
            f'<div class="ff-pilihan"><div class="ff-pilihan-film">{poster(judul, poster_path, 112, 168)}'
            f'<div class="ff-pilihan-info"><div class="ff-pilihan-judul">{esc(judul)}</div>'
            f'<div class="ff-chips">{"".join(chips)}</div>{sin}</div></div>'
            f'<div class="ff-garis-v"></div>'
            f'<div class="ff-pilihan-prm"><div class="ff-prm-judul">Parameter Rekomendasi</div>{prm}</div></div></div>')


def kartu_metrik(nama_ikon: str, warna: str, nilai: str, label: str, seri: Sequence[float] | None = None, id_: str = "m") -> str:
    """Kartu metrik: ikon berwarna, angka besar, label, dan garis mini bergradien."""
    return (f'<div class="ff-metrik"><div class="ff-metrik-atas"><span class="ff-metrik-ikon" style="--w:{warna}">{ikon(nama_ikon, warna, 22, 1.8)}</span>'
            f'<div><div class="ff-metrik-nilai">{esc(nilai)}</div><div class="ff-metrik-label">{esc(label)}</div></div></div>'
            f'{sparkline(seri, warna, id_=id_)}</div>')


def baris_metrik(kartu: Sequence[str]) -> str:
    return f'<div class="ff-baris-metrik">{"".join(kartu)}</div>'


def tabel_rekomendasi(baris: Sequence[dict]) -> str:
    """Tabel HTML: #, Judul (thumbnail + judul), Tahun, Genre, Skor Kemiripan (angka + bar merah).

    ``baris``: dict dengan kunci ``no, judul, tahun, genre, skor, poster_path``.
    """
    kepala = ("<tr><th class='c'>#</th><th>Judul Film</th><th class='c'>Tahun</th><th>Genre</th>"
              "<th>Skor Kemiripan</th></tr>")
    isi = ""
    for b in baris:
        skor = max(0.0, min(1.0, float(b["skor"])))
        isi += (f'<tr><td class="c ff-no">{esc(b["no"])}</td>'
                f'<td><div class="ff-judul-sel">{poster(b["judul"], b.get("poster_path"), 30, 44)}<span>{esc(b["judul"])}</span></div></td>'
                f'<td class="c">{esc(b["tahun"])}</td><td class="ff-genre-sel">{esc(b["genre"])}</td>'
                f'<td><div class="ff-skor"><b>{skor:.3f}</b><div class="ff-bar"><i style="width:{skor * 100:.1f}%"></i></div></div></td></tr>')
    return f'<table class="ff-tabel"><thead>{kepala}</thead><tbody>{isi}</tbody></table>'


def kartu_langkah(nomor: str, judul: str, isi: str, nama_ikon: str) -> str:
    """Kartu alur singkat di Beranda."""
    return (f'<div class="ff-langkah"><div class="ff-langkah-ikon">{ikon(nama_ikon, MERAH, 22, 1.8)}</div>'
            f'<div class="ff-langkah-no">{esc(nomor)}</div><h4>{esc(judul)}</h4><p>{esc(isi)}</p></div>')
