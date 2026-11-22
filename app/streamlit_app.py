"""
Mix It Up - clickable demo of the recommender.

Run from the project folder:
    streamlit run app/streamlit_app.py
"""
import sys
import urllib.parse
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import AUDIO_COLS, build_features, load_songs  # noqa: E402
from evaluate import PERSONAS, make_history, metrics  # noqa: E402
from recommender import BaselineRecommender, MixItUp  # noqa: E402

st.set_page_config(page_title="Mix It Up", page_icon=":material/shuffle:", layout="centered")

CARD_COLOURS = [("#7b5cff", "#00c6ff"), ("#ff512f", "#dd2476"), ("#11998e", "#38ef7d"),
                ("#f7971e", "#ff5e62"), ("#c471f5", "#fa71cd"), ("#2193b0", "#6dd5ed"),
                ("#ee9ca7", "#ffdde1")]


# ---------------------------------------------------------------- engine ---
@st.cache_resource(show_spinner="Training the recommender...")
def load_engine():
    df = load_songs()
    X, info = build_features(df)
    mix = MixItUp(df, X)
    base = BaselineRecommender(df, X)
    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    return df, X, info, mix, base, xy


df, X, info, mix, base, XY = load_engine()


# ----------------------------------------------------------------- state ---
def new_mix():
    s = st.session_state
    s.seed += 1
    s.mix = list(mix.recommend(s.history, leave_out=s.leave_out, surprise=s.surprise,
                               liked=s.saved, disliked=s.disliked, seed=s.seed)["idx"])
    s.revealed, s.current = set(), None


def reset_user():
    s = st.session_state
    s.history = make_history(df, PERSONAS[s.persona], seed=list(PERSONAS).index(s.persona))
    s.saved, s.disliked, s.seed = [], [], 0
    new_mix()


if "persona" not in st.session_state:
    st.session_state.update(persona=list(PERSONAS)[0], surprise=0.5,
                            leave_out={"genres": [], "artists": [], "topics": []})
    reset_user()
    if st.query_params.get("demo") == "play":          # used for documentation screenshots
        st.session_state.current = 0
        st.session_state.revealed = {0, 1}
s = st.session_state


def reveal(i):
    s.current = i
    s.revealed.add(i)


def next_song():
    hidden = [i for i in range(len(s.mix)) if i not in s.revealed]
    if hidden:
        reveal(hidden[0])


def save(song):
    if song not in s.saved:
        s.saved.append(song)
    st.toast("Saved to your library")


def not_for_me(song):
    if song not in s.disliked:
        s.disliked.append(song)
    next_song()


# ------------------------------------------------------------------- css ---
css = """
<style>
.block-container {max-width: 760px; padding-top: 2rem;}
.hero {padding: 20px 22px; border-radius: 20px;
       background: linear-gradient(135deg,#5b2bd9 0%,#c2307a 60%,#ff7a45 100%);
       box-shadow: 0 8px 24px rgba(194,48,122,.35); color: #fff; margin-bottom: 12px;}
.pill {display:inline-block; padding:3px 10px; border-radius:13px; background:rgba(255,255,255,.2);
       font-size:12px; font-weight:700; letter-spacing:.4px;}
.hero h2 {font-size:26px; line-height:32px; margin:12px 0 6px; color:#fff; padding:0}
.hero p {font-size:14px; margin:0; color:rgba(255,255,255,.92)}
.steps {display:flex; gap:8px; margin:8px 0 4px; text-align:center}
.steps div {flex:1; font-size:13px; color:#ddd}
.steps b {display:inline-flex; width:30px; height:30px; border-radius:50%; align-items:center;
          justify-content:center; color:#0f0f0f; margin-bottom:6px}
.art {height:190px; border-radius:12px; display:flex; align-items:center; justify-content:center;
      font-size:64px; color:rgba(255,255,255,.5)}
.meta {color:#aaa; font-size:14px}
.segments {display:flex; gap:6px; margin:6px 0 2px}
.segments span {flex:1; height:4px; border-radius:2px; background:#4d4d4d}
.segments span.on {background:#fff}
.empty {padding:28px 20px; border-radius:14px; background:#212121; text-align:center}
"""
for i, (a, b) in enumerate(CARD_COLOURS):
    css += f""".st-key-card_{i} button {{width:76px; height:76px; border-radius:50%; border:none;
        background:linear-gradient(140deg,{a},{b}); box-shadow:0 6px 18px {a}66; margin:auto;}}
        .st-key-card_{i} button p {{font-size:30px; font-weight:700; color:#fff}}
        .st-key-card_{i} button:hover {{transform:scale(1.06); color:#fff}}\n"""
st.markdown(css + "</style>", unsafe_allow_html=True)

# --------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.subheader("Who is listening?")
    st.selectbox("Simulated student", list(PERSONAS), key="persona", on_change=reset_user)
    with st.expander("Listening history"):
        hist = df.loc[list(s.history)].assign(plays=list(s.history.values()))
        st.dataframe(hist.groupby(["artist", "genre"])["plays"].sum()
                     .sort_values(ascending=False).reset_index(), hide_index=True)
    st.slider("Surprise level", 0.0, 1.0, key="surprise", step=0.1,
              help="0 = close to what you know, 1 = far outside your bubble")
    st.caption("Saved songs and 'Not for me' are used the next time you press Mix again.")

view = st.segmented_control("View", ["Mix It Up", "Library", "How it works"],
                            default=st.query_params.get("view", "Mix It Up"),
                            label_visibility="collapsed") or "Mix It Up"

# ---------------------------------------------------------- Mix It Up ---
if view == "Mix It Up":
    st.markdown("""<div class="hero"><span class="pill">NEW EVERY TIME</span>
        <h2>Seven songs you've never heard.</h2>
        <p>Mix It Up skips your usual rotation and hands you a surprise set from artists and
        genres outside your comfort zone. Open a card, hear it, keep what you love.</p></div>
        <div class="steps"><div><b style="background:#ffd166">1</b><br>Tap a card to reveal a song</div>
        <div><b style="background:#6ee7b7">2</b><br>Listen and skip freely</div>
        <div><b style="background:#93c5fd">3</b><br>Save your favourites</div></div>""",
                unsafe_allow_html=True)

    if not s.mix:
        st.markdown('<div class="empty">No new songs match your filters. '
                    'Remove a filter to see more.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='text-align:center;color:#ddd;margin:14px 0 6px'>"
                    f"Your {len(s.mix)} mystery songs</p>", unsafe_allow_html=True)
        cols = st.columns(len(s.mix))
        for i, col in enumerate(cols):
            label = "▶" if s.current == i else ("✓" if i in s.revealed else "?")
            col.button(label, key=f"card_{i}", on_click=reveal, args=(i,),
                       help=f"Mystery song {i + 1}")

    left, right = st.columns(2)
    with left.popover("Leave out", icon=":material/filter_list:", width="stretch"):
        st.caption("Pick artists or genres you don't want in your mix.")
        lo = s.leave_out
        lo["genres"] = st.multiselect("Genres", info["genres"], default=lo["genres"])
        lo["topics"] = st.multiselect("Lyric topics", sorted(df["topic"].unique()), default=lo["topics"])
        lo["artists"] = st.multiselect("Artists", sorted(df["artist"].unique()), default=lo["artists"])
        st.button("Done", on_click=new_mix, type="primary")
    right.button("Play now", type="primary", width="stretch",
                 on_click=next_song, disabled=not s.mix)

    if s.current is not None and s.mix:
        song = s.mix[s.current]
        row = df.loc[song]
        why = mix.explain(song, s.history)
        a, b = CARD_COLOURS[s.current]
        st.divider()
        st.markdown(f'<div class="art" style="background:linear-gradient(135deg,{a},{b})">♪</div>',
                    unsafe_allow_html=True)
        st.subheader(row["title"])
        st.markdown(f"**{row['artist']}**  \n<span class='meta'>{row['genre']} · {row['year']} · "
                    f"{mix.cluster_names[mix.labels[song]]}</span>", unsafe_allow_html=True)
        st.caption(why["short"])
        segs = "".join(f"<span class='{'on' if i in s.revealed else ''}'></span>" for i in range(len(s.mix)))
        st.markdown(f"<span class='meta'>Song {s.current + 1} of {len(s.mix)}</span>"
                    f"<div class='segments'>{segs}</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.button("Save to library", icon=":material/bookmark:", on_click=save, args=(song,),
                  width="stretch", disabled=song in s.saved)
        c2.button("Not for me", icon=":material/close:", on_click=not_for_me, args=(song,),
                  width="stretch")
        c3.button("Next", icon=":material/skip_next:", on_click=next_song, width="stretch",
                  disabled=len(s.revealed) == len(s.mix))
        query = urllib.parse.quote(f"{row['artist']} {row['title']}")
        st.link_button("Listen on YouTube Music", f"https://music.youtube.com/search?q={query}",
                       icon=":material/open_in_new:")

        with st.expander("Why this song?", expanded=st.query_params.get("demo") == "play"):
            st.write(why["long"])
            st.caption(f"Closest song in your history: {why['anchor']}")
            bars = pd.DataFrame({"feature": AUDIO_COLS * 2,
                                 "value": list(why["song_features"].values()) + list(why["anchor_features"].values()),
                                 "song": ["This song"] * 6 + ["Your song"] * 6})
            st.altair_chart(alt.Chart(bars).mark_bar().encode(
                x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                y=alt.Y("song:N", title=None, axis=None),
                row=alt.Row("feature:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
                color=alt.Color("song:N", scale=alt.Scale(range=["#c2307a", "#9e9e9e"]), title=None),
            ).properties(height=34, width=420), width="content")

    st.divider()
    st.button("Mix again", icon=":material/shuffle:", on_click=new_mix, width="stretch")

# ------------------------------------------------------------ Library ---
elif view == "Library":
    st.subheader("Your library")
    if not s.saved:
        st.info("Nothing saved yet. Open a mystery card and press Save to library.")
    else:
        lib = df.loc[s.saved, ["title", "artist", "genre", "year"]]
        st.dataframe(lib, hide_index=True, width="stretch")
    if s.disliked:
        st.caption(f"{len(s.disliked)} songs marked as Not for me - the next mix moves away from them.")

# ------------------------------------------------------- How it works ---
else:
    st.subheader("How Mix It Up picks your songs")
    st.markdown("""
1. **Song vectors** - every song becomes numbers: audio features, genre, lyric topics and lyrics (TF-IDF + SVD).
2. **Taste regions** - K-Means groups the 10,000+ songs into 10 regions (the colours below).
3. **Your bubble** - the regions that hold at least 15% of your plays.
4. **Seven picks** - new artists only, Leave-out filters applied, part of the mix from regions just outside your bubble, MMR keeps the seven songs different from each other.
""")
    rng = np.random.default_rng(0)
    sample = rng.choice(len(df), 2500, replace=False)
    pts = pd.DataFrame({"x": XY[sample, 0], "y": XY[sample, 1],
                        "region": [mix.cluster_names[mix.labels[i]] for i in sample], "kind": "catalogue"})
    you = pd.DataFrame({"x": XY[list(s.history), 0], "y": XY[list(s.history), 1], "kind": "your history"})
    picks = pd.DataFrame({"x": XY[s.mix, 0], "y": XY[s.mix, 1], "kind": "this mix",
                          "title": df.loc[s.mix, "title"].values}) if s.mix else pd.DataFrame(columns=["x", "y"])
    xy = dict(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None))
    layer = alt.Chart(pts).mark_circle(size=14, opacity=.5).encode(
        **xy, color=alt.Color("region:N", legend=alt.Legend(orient="bottom", columns=2)))
    layer += alt.Chart(you).mark_circle(size=70, color="white", opacity=.9).encode(**xy)
    if s.mix:
        layer += alt.Chart(picks).mark_point(shape="diamond", size=260, filled=True, color="#ff7a45",
                                             stroke="black", opacity=1).encode(**xy, tooltip="title")
    st.altair_chart(layer.properties(height=420), width="stretch")
    st.caption("White dots = your history, orange diamonds = this mix. Map made with PCA.")

    if s.mix:
        b = base.recommend(s.history)
        m = df.loc[s.mix].assign(idx=s.mix)
        comp = pd.DataFrame({"Today's recommender (k-NN)": metrics(b, s.history, X, df, mix),
                             "Mix It Up": metrics(m, s.history, X, df, mix)}).T.round(2)
        st.markdown("**Compared with a classic 'more of the same' recommender**")
        st.dataframe(comp, width="stretch")
