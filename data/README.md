# Data

`tcc_ceds_music.csv` - *Music Dataset: Lyrics and Metadata from 1950 to 2019*
by Moura, Fontelles, Sampaio & França (2020), Mendeley Data V3,
https://doi.org/10.17632/3t9vbwxgr5.3, also on
[Kaggle](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019).
License: CC BY 4.0.

28,372 songs, one row per song:

| Column(s) | Meaning |
|---|---|
| `artist_name`, `track_name`, `release_date`, `genre` | metadata (7 genres) |
| `lyrics` | pre-processed lyrics (stop words removed, lemmatised) |
| `danceability`, `loudness`, `acousticness`, `instrumentalness`, `valence`, `energy` | audio features (Spotify / Echo Nest), scaled 0-1 |
| `dating` ... `feelings` (16 columns) | lyric-topic scores (LDA), 0-1 |
| `topic` | the strongest lyric topic |

The project uses the 10,412 songs released from 2000 onward.
