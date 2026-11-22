"""
data.py - load the song catalogue and turn every song into a vector.

Step 1 of the pipeline:
    CSV  ->  clean DataFrame  ->  song vectors (one row per song)

Each song vector is built from four "blocks" of information:
    1. audio   - how the song sounds (danceability, energy, ...)
    2. genre   - one-hot encoded genre
    3. topics  - what the lyrics are about (romantic, sadness, ...)
    4. lyrics  - a compressed summary of the actual words (TF-IDF + SVD)

Every block is normalised to the same length, multiplied by a weight and
then glued together. Because all vectors end up with length 1, the dot
product of two songs equals their cosine similarity (1 = identical, 0 =
unrelated). The rest of the project relies on that.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tcc_ceds_music.csv"

AUDIO_COLS = ["danceability", "loudness", "acousticness",
              "instrumentalness", "valence", "energy"]

TOPIC_COLS = ["dating", "violence", "world/life", "night/time",
              "shake the audience", "family/gospel", "romantic",
              "communication", "obscene", "music", "movement/places",
              "light/visual perceptions", "family/spiritual", "like/girls",
              "sadness", "feelings"]

# How much each block counts when we compare two songs.
BLOCK_WEIGHTS = {"audio": 1.0, "genre": 1.0, "topics": 0.6, "lyrics": 0.6}


def load_songs(path=DATA_PATH, min_year=2000):
    """Read the Kaggle CSV and keep the songs a student today would hear."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df[df["release_date"] >= min_year]
    df = df.drop_duplicates(subset=["artist_name", "track_name"])
    df = df.dropna(subset=["lyrics"]).reset_index(drop=True)

    # Nicer display names ("bruno mars" -> "Bruno Mars")
    df["artist"] = df["artist_name"].str.title()
    df["title"] = df["track_name"].str.title()
    df["year"] = df["release_date"]
    return df


def _block(matrix):
    """Give every row length 1 so that no block dominates the others."""
    return normalize(np.asarray(matrix, dtype=float))


def build_features(df, lyric_dims=24, seed=42):
    """Turn the DataFrame into a matrix X with one unit-length row per song.

    Returns X and a dict with the fitted parts (handy for the notebook).
    """
    # 1. Audio: put every feature on the same scale (mean 0, std 1)
    audio = StandardScaler().fit_transform(df[AUDIO_COLS])

    # 2. Genre: one column per genre with a 1 for the song's genre
    genre = pd.get_dummies(df["genre"]).to_numpy(dtype=float)

    # 3. Lyric topics: already scores between 0 and 1 in the dataset
    topics = StandardScaler().fit_transform(df[TOPIC_COLS])

    # 4. Lyrics: TF-IDF finds words that are typical for a song,
    #    Truncated SVD (Latent Semantic Analysis) squeezes ~5000 word
    #    columns into a few "themes".
    tfidf = TfidfVectorizer(max_features=5000, min_df=3, stop_words="english")
    words = tfidf.fit_transform(df["lyrics"])
    svd = TruncatedSVD(n_components=lyric_dims, random_state=seed)
    lyrics = svd.fit_transform(words)

    blocks = {"audio": audio, "genre": genre, "topics": topics, "lyrics": lyrics}
    X = np.hstack([_block(m) * BLOCK_WEIGHTS[name] for name, m in blocks.items()])
    X = normalize(X)  # final vector length 1 -> dot product = cosine

    info = {"tfidf": tfidf, "svd": svd, "blocks": blocks,
            "genres": sorted(df["genre"].unique())}
    return X, info
