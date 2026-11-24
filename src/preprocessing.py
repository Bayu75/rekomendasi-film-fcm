
import ast
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# Lokasi folder proyek
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_preprocessing_artifacts(path=None):
    """
    Memuat artefak preprocessing yang sudah dilatih di Google Colab.
    """
    if path is None:
        path = DATA_DIR / "preprocessing_artifacts.pkl"

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File artefak preprocessing tidak ditemukan: {path}"
        )

    artifacts = joblib.load(path)

    return artifacts


def transform_new(data, artifacts):
    """
    Mengubah data film menjadi 14 fitur menggunakan
    scaler dan PCA yang sudah dilatih sebelumnya.
    """

    data = data.copy()

    numeric_features = artifacts["numeric_features"]
    valid_genres = artifacts["valid_genres"]

    scaler_numeric = artifacts["scaler_numeric"]
    pca_genre = artifacts["pca_genre"]
    scaler_genre_pca = artifacts["scaler_genre_pca"]

    # 1. Validasi kolom yang dibutuhkan
    required_columns = numeric_features + ["genres"]
    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {missing_columns}"
        )

    
    # Validasi nilai numerik
    if data[numeric_features].isnull().any().any():
        raise ValueError("Data numerik mengandung missing value.")

    if not data[numeric_features].apply(
        pd.to_numeric, errors="coerce"
    ).notnull().all().all():
        raise ValueError("Data numerik harus berisi angka.")

    if (data[["vote_count", "popularity"]] < 0).any().any():
        raise ValueError(
            "vote_count dan popularity tidak boleh negatif."
        )

    # 2. Transformasi fitur numerik
    X_numeric = data[numeric_features].copy()

    X_numeric["vote_count"] = np.log1p(X_numeric["vote_count"])
    X_numeric["popularity"] = np.log1p(X_numeric["popularity"])

    X_numeric_scaled = pd.DataFrame(
        scaler_numeric.transform(X_numeric),
        columns=numeric_features,
        index=data.index
    )


    # Validasi data genre
    if data["genres"].isnull().any():
        raise ValueError("Kolom genres mengandung missing value.")
    

    # 3. One-Hot Encoding genre
    genre_encoded = pd.DataFrame(
        0,
        index=data.index,
        columns=valid_genres,
        dtype=int
    )

    for idx, genres in data["genres"].items():

        if isinstance(genres, str):
            genre_list = ast.literal_eval(genres)
        else:
            genre_list = genres

        for genre in genre_list:
            if isinstance(genre, dict):
                genre_name = genre["name"]
            else:
                genre_name = genre

            if genre_name in valid_genres:
                genre_encoded.loc[idx, genre_name] = 1

    # 4. Transformasi PCA genre
    genre_pca = pca_genre.transform(genre_encoded)

    genre_pca_df = pd.DataFrame(
        genre_pca,
        index=data.index,
        columns=[
            f"Genre_PC{i+1}"
            for i in range(pca_genre.n_components_)
        ]
    )

    # 5. Scaling hasil PCA
    genre_pca_scaled = pd.DataFrame(
        scaler_genre_pca.transform(genre_pca_df),
        index=data.index,
        columns=genre_pca_df.columns
    )

    # 6. Gabungkan fitur numerik dan genre PCA
    X_final = pd.concat(
        [X_numeric_scaled, genre_pca_scaled],
        axis=1
    )

    # 7. Samakan urutan kolom dengan data training
    X_final = X_final[artifacts["feature_columns"]]

    return X_final

# ---------------------------------------------------------------------------
# Fungsi kontrak PRD untuk fitur upload CSV
# ---------------------------------------------------------------------------

KOLOM_WAJIB_CSV = ["id", "title", "genres", "vote_average", "vote_count",
                   "popularity", "runtime", "release_date"]


def _daftar_genre(genres):
    """Ubah isi kolom genres (string/list) menjadi list nama genre."""
    if isinstance(genres, str):
        try:
            genres = ast.literal_eval(genres)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(genres, (list, tuple)):
        return []
    return [g["name"] if isinstance(g, dict) else str(g) for g in genres]


def validate_csv(df):
    """Validasi CSV unggahan. Return list pesan error (kosong = valid)."""
    pesan = []
    hilang = [k for k in KOLOM_WAJIB_CSV if k not in df.columns]
    if hilang:
        return [f"Kolom berikut tidak ditemukan: {', '.join(hilang)}."]
    if len(df) == 0:
        return ["File tidak berisi baris data."]
    for k in ["vote_average", "vote_count", "popularity", "runtime"]:
        angka = pd.to_numeric(df[k], errors="coerce")
        rusak = int(angka.isna().sum() - df[k].isna().sum())
        if rusak > 0:
            pesan.append(f"Kolom {k} berisi {rusak} nilai yang bukan angka.")
    if pd.to_datetime(df["release_date"], errors="coerce").notna().sum() == 0:
        pesan.append("Kolom release_date tidak memiliki tanggal valid (format YYYY-MM-DD).")
    return pesan


def load_and_clean(path):
    """Baca CSV mentah lalu bersihkan: tipe data, missing value, duplikat."""
    df = pd.read_csv(path, low_memory=False)
    for k in ["vote_average", "vote_count", "popularity", "runtime"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year
    df = df.dropna(subset=["title", "vote_average", "vote_count",
                           "popularity", "runtime", "release_year"])
    df = df[(df["vote_count"] >= 0) & (df["popularity"] >= 0) & (df["runtime"] > 0)]
    df = df[df["genres"].apply(lambda g: len(_daftar_genre(g)) > 0)]
    return df.drop_duplicates(subset="id").reset_index(drop=True)


def build_features(df, artifacts=None):
    """Bentuk X (n, 14) memakai artefak preprocessing yang sudah dilatih.

    Return: X (np.ndarray), df_film (DataFrame), artefak (dict).
    """
    if artifacts is None:
        artifacts = load_preprocessing_artifacts()
    df = df.reset_index(drop=True)
    X = transform_new(df, artifacts).to_numpy(dtype=float)
    X = np.clip(X, 0.0, 1.0)

    df_film = pd.DataFrame({
        "id": df["id"],
        "title": df["title"].astype(str),
        "genres": df["genres"].apply(_daftar_genre),
        "vote_average": df["vote_average"].astype(float),
        "vote_count": df["vote_count"].astype(float),
        "popularity": df["popularity"].astype(float),
        "runtime": df["runtime"].astype(float),
        "year": df["release_year"].astype(int),
    })

    artefak = {"prep": artifacts,
               "feature_names": list(artifacts["feature_columns"])}
    pca_path = PROJECT_ROOT / "models" / "pca_fcm.joblib"
    if pca_path.exists():
        pca_fcm = joblib.load(pca_path)
        artefak["pca_fcm"] = pca_fcm
        artefak["pca_2d"] = pca_fcm
    return X, df_film, artefak
    
# Alias agar provider.py yang masih memakai nama lama tetap jalan
transform_new_data = transform_new