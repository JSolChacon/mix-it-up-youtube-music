"""
evaluate.py - simulated listeners and the metrics we use to compare
the baseline with Mix It Up.

The Kaggle dataset has no users, so we simulate students. Real people
listen to *artists*, not random songs, so a simulated history is built by
picking a handful of artists from the persona's favourite genres and
playing their songs a random number of times.
"""
import itertools

import numpy as np

from recommender import taste_vector

PERSONAS = {
    "Sam - pop & hip hop":    {"pop": 0.6, "hip hop": 0.4},
    "Lotte - indie rock":     {"rock": 0.7, "blues": 0.3},
    "Noah - chill reggae":    {"reggae": 0.6, "pop": 0.4},
    "Mila - jazz & soul":     {"jazz": 0.6, "blues": 0.4},
    "Daan - country roads":   {"country": 0.75, "rock": 0.25},
}


def make_history(df, genre_mix, n_artists=8, songs_per_artist=5, seed=0):
    """Return {song_index: plays} for a simulated student."""
    rng = np.random.default_rng(seed)
    counts = df.groupby("artist").size()
    history = {}
    genres = list(genre_mix)
    weights = np.array(list(genre_mix.values()))
    chosen = set()
    while len(chosen) < n_artists:
        g = rng.choice(genres, p=weights / weights.sum())
        pool = df[(df["genre"] == g) & df["artist"].isin(counts[counts >= 3].index)]
        artist = rng.choice(pool["artist"].unique())
        if artist in chosen:
            continue
        chosen.add(artist)
        songs = df.index[df["artist"] == artist].to_numpy()
        for s in rng.choice(songs, size=min(songs_per_artist, len(songs)), replace=False):
            history[int(s)] = int(rng.integers(1, 12))
    return history


def metrics(recs, history, X, df, engine):
    """Beyond-accuracy metrics for one list of recommendations."""
    ids = list(recs["idx"])
    v = taste_vector(X, history)               # same yardstick for both models
    sims = X[ids] @ X[ids].T
    pairs = [sims[i, j] for i, j in itertools.combinations(range(len(ids)), 2)]
    known_artists = set(df.loc[list(history), "artist"])
    core = set(engine.taste_clusters(history))
    return {
        "relevance": float(np.mean(X[ids] @ v)),                     # fits your taste?
        "diversity": float(1 - np.mean(pairs)),                      # differ from each other?
        "genres": int(recs["genre"].nunique()),                      # how many genres?
        "new_artists": float(np.mean(~recs["artist"].isin(known_artists))),
        "outside_bubble": float(np.mean([engine.labels[i] not in core for i in ids])),
    }
