
import pandas as pd

from src.preprocessing import (
    load_preprocessing_artifacts,
    transform_new
)


def main():
    # 1. Memuat artefak preprocessing
    artifacts = load_preprocessing_artifacts()
    print("Artefak preprocessing berhasil dimuat!")

    # 2. Memuat dataset film
    data_path = "data/processed/movies_final.csv"
    movies = pd.read_csv(data_path)

    # 3. Menguji transformasi pada 5 film
    data_uji = movies.head(5).copy()

    # Menguji validasi nilai numerik
    data_invalid = data_uji.copy()
    data_invalid.loc[data_invalid.index[0], "vote_count"] = -10

    try:
        transform_new(data_invalid, artifacts)
    except ValueError as e:
        print("\nValidasi berhasil mendeteksi kesalahan:")
        print(e)

    # Menguji transformasi data normal
    hasil = transform_new(data_uji, artifacts)

    print("\n=== PENGUJIAN TRANSFORMASI ===")
    print("Ukuran hasil:", hasil.shape)
    print("Jumlah fitur:", hasil.shape[1])
    print("Missing value:", hasil.isnull().sum().sum())

    # 4. Validasi dataset hasil preprocessing
    X_final = pd.read_csv("data/processed/X_final.csv")
    G = pd.read_csv("data/processed/G.csv")

    print("\n=== VALIDASI DATASET ===")
    print("X_final:", X_final.shape)
    print("movies_final:", movies.shape)
    print("G:", G.shape)

    print("\nJumlah ID film unik:", movies["id"].nunique())
    print(
        "Jumlah baris sama:",
        len(X_final) == len(movies) == len(G)
    )

    print("\n=== STATUS MISSING VALUE ===")
    print("X_final:", X_final.isnull().sum().sum())
    print("G:", G.isnull().sum().sum())

    print("\nKolom movies_final yang memiliki missing value:")
    missing = movies.isnull().sum()
    print(missing[missing > 0])

    print("\nPengujian selesai!")


if __name__ == "__main__":
    main()
