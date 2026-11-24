
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
