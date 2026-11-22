"""
recommender.py - two recommenders that work on the song vectors from data.py.

A listening history is a dict {song_index: number_of_plays}.

BaselineRecommender  - our re-creation of how YouTube Music behaves today:
                       "give me the songs closest to everything you played".
                       Accurate, but it keeps you inside your bubble.

MixItUp              - the extension: seven songs you have never heard,
                       still based on your taste but deliberately outside
                       your usual rotation, with filters and explanations.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from data import AUDIO_COLS


def taste_vector(X, history):
    """The user's taste = the play-weighted average of the songs they played."""
    ids = np.array(list(history.keys()))
    plays = np.array(list(history.values()), dtype=float)
    v = (X[ids] * plays[:, None]).sum(axis=0)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# 1. Baseline: the "bubble" recommender
# ---------------------------------------------------------------------------
class BaselineRecommender:
    """k-Nearest Neighbours on cosine similarity (content-based filtering)."""

    def __init__(self, df, X):
        self.df, self.X = df, X
        self.knn = NearestNeighbors(metric="cosine").fit(X)

    def recommend(self, history, n=7):
        v = taste_vector(self.X, history)
        # ask for extra neighbours because songs already played are skipped
        _, idx = self.knn.kneighbors([v], n_neighbors=n + len(history))
        picks = [i for i in idx[0] if i not in history][:n]
        return self.df.loc[picks].assign(idx=picks)


# ---------------------------------------------------------------------------
# 2. Mix It Up: discovery outside the bubble
# ---------------------------------------------------------------------------
class MixItUp:
    """Discovery recommender built in five steps:

    1. K-Means groups all songs into "taste regions" (clusters).
    2. A robust taste profile only uses regions the user really listens to,
       so one odd song does not pollute the recommendations.
    3. Candidate filter: never-heard artists, nothing too similar to what the
       user already knows, nothing from the Leave-out list.
    4. Comfort-zone quota + Maximal Marginal Relevance (MMR): part of the mix
       comes from neighbouring regions outside the bubble, and every next
       pick must be relevant AND different from the picks before it.
    5. Every pick gets a human explanation ("Why this song?").
    """

    def __init__(self, df, X, n_clusters=10, seed=42):
        self.df, self.X = df, X
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10,
                             random_state=seed).fit(X)
        self.labels = self.kmeans.labels_
        self.cluster_names = self._name_clusters()

    # -- step 1: give every cluster a readable name -------------------------
    def _name_clusters(self):
        friendly = {"obscene": "explicit", "world/life": "life", "night/time": "nightlife"}
        names = {}
        median_energy = self.df["energy"].median()
        for c in np.unique(self.labels):
            part = self.df[self.labels == c]
            mood = "upbeat" if part["energy"].mean() > median_energy else "mellow"
            topic = part["topic"].mode()[0]
            names[int(c)] = f"{mood} {part['genre'].mode()[0]} · {friendly.get(topic, topic)}"
        return names

    # -- step 2: robust taste profile ---------------------------------------
    def taste_clusters(self, history, min_share=0.15):
        """Clusters that hold at least `min_share` of all plays."""
        plays = pd.Series(history, dtype=float)
        share = plays.groupby(self.labels[plays.index]).sum() / plays.sum()
        core = share[share >= min_share].index.tolist()
        return core or [share.idxmax()]

    def profile(self, history, liked=(), disliked=(), min_share=0.15):
        core = set(self.taste_clusters(history, min_share))
        core_history = {i: p for i, p in history.items() if self.labels[i] in core}
        v = taste_vector(self.X, core_history)
        # Relevance feedback (Rocchio): move towards saved songs,
        # away from songs marked "Not for me".
        if liked:
            v = v + 0.5 * self.X[list(liked)].mean(axis=0)
        if disliked:
            v = v - 0.3 * self.X[list(disliked)].mean(axis=0)
        return v / np.linalg.norm(v)

    # -- steps 3 + 4: filter and re-rank ------------------------------------
    def recommend(self, history, n=7, leave_out=None, surprise=0.5,
                  liked=(), disliked=(), seed=None, pool_size=250,
                  too_close=0.92):
        df, X = self.df, self.X
        leave_out = leave_out or {}
        v = self.profile(history, liked, disliked)
        relevance = X @ v

        # Step 3: which songs are allowed?
        heard = set(history) | set(liked) | set(disliked)
        known_artists = set(df.loc[list(heard), "artist"])
        ok = ~df.index.isin(list(heard))
        ok &= ~df["artist"].isin(known_artists)                   # new artists only
        ok &= ~df["genre"].isin(leave_out.get("genres", []))      # Leave out
        ok &= ~df["artist"].isin(leave_out.get("artists", []))
        ok &= ~df["topic"].isin(leave_out.get("topics", []))
        sim_to_history = X @ X[list(history)].T
        ok &= sim_to_history.max(axis=1) < too_close              # not "more of the same"

        candidates = np.where(ok)[0]
        if len(candidates) == 0:
            return df.iloc[[]].assign(idx=[])                     # empty state (screen B-03e)

        # Comfort-zone quota: part of the mix MUST come from taste regions
        # the user does not listen to yet (the closest ones = "one step out").
        core = set(self.taste_clusters(history))
        outside = np.array([self.labels[i] not in core for i in candidates])
        n_out = int(round(n * surprise))

        rng = np.random.default_rng(seed)                         # "New every time"
        picks, artists = [], set()
        for group, k in ((candidates[outside], n_out),
                         (candidates[~outside], n - n_out)):
            pool = group[np.argsort(-relevance[group])][:pool_size]
            noisy = relevance[pool] + rng.gumbel(0, 0.02, size=len(pool))
            self._mmr(pool, noisy, picks, artists, k, lam=1.0 - 0.6 * surprise)
        if len(picks) < n:                                        # top up if a group ran dry
            pool = candidates[np.argsort(-relevance[candidates])][:pool_size]
            self._mmr(pool, relevance[pool], picks, artists, n - len(picks), lam=0.7)
        rng.shuffle(picks)                                        # hide which is which

        out = df.loc[picks].assign(idx=picks, match=relevance[picks])
        out["region"] = [self.cluster_names[self.labels[i]] for i in picks]
        out["reason"] = [self.explain(i, history)["short"] for i in picks]
        return out

    def _mmr(self, pool, scores, picks, artists, k, lam):
        """Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

        Repeatedly take the song with the best mix of
            lam       * relevance              (fits your taste)
          - (1 - lam) * similarity to picks    (but is not a copy of a pick)
        Max. one song per artist. Adds to `picks` in place.
        """
        X, df = self.X, self.df
        added = 0
        while added < k:
            if picks:
                redundancy = (X[pool] @ X[picks].T).max(axis=1)
            else:
                redundancy = np.zeros(len(pool))
            score = lam * scores - (1 - lam) * redundancy
            for j in np.argsort(-score):
                song = int(pool[j])
                if song not in picks and df.at[song, "artist"] not in artists:
                    picks.append(song)
                    artists.add(df.at[song, "artist"])
                    added += 1
                    break
            else:
                return                                            # pool exhausted

    # -- step 5: explanation -------------------------------------------------
    def explain(self, song, history):
        """Find the song in the history that is closest and say what they share."""
        df, X = self.df, self.X
        hist = np.array(list(history))
        anchor = hist[np.argmax(X[hist] @ X[song])]
        a, s = df.loc[anchor], df.loc[song]

        # Only talk about features a listener can hear and that are present
        # (two songs both having ~0 instrumentalness is not a real reason).
        audible = [c for c in ("energy", "danceability", "valence", "acousticness", "loudness")
                   if max(a[c], s[c]) > 0.15]
        diffs = {c: abs(a[c] - s[c]) for c in audible}
        shared = (sorted(diffs, key=diffs.get) + ["energy", "danceability"])[:2]
        if a["genre"] == s["genre"]:
            new = f"a new artist in {s['genre']}"
        else:
            new = f"a step from {a['genre']} into {s['genre']}"
        return {
            "anchor": f"{a['title']} - {a['artist']}",
            "short": f"Because you play {a['artist']}",
            "long": (f"Because you play {a['artist']}: similar {shared[0]} and "
                 f"{shared[1]}, but {new}."),
            "shared": shared,
            "song_features": s[AUDIO_COLS].to_dict(),
            "anchor_features": a[AUDIO_COLS].to_dict(),
        }
