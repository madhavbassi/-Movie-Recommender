"""Build reproducible CineMatch model artifacts from the movie metadata CSV."""

import argparse
import ast
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parent


def parse_genres(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value) if value else []
        return [item["name"] for item in parsed if isinstance(item, dict) and item.get("name")]
    except (ValueError, SyntaxError, TypeError):
        return []


def build(source: Path, output_dir: Path) -> None:
    raw = pd.read_csv(source, low_memory=False)
    columns = [
        "id", "title", "overview", "tagline", "genres", "vote_average",
        "vote_count", "popularity", "release_date", "original_language", "poster_path",
    ]
    movies = raw[columns].copy()
    movies["movie_id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["movie_id", "title"]).drop_duplicates("movie_id")
    movies["movie_id"] = movies["movie_id"].astype(int)
    movies["genres"] = movies["genres"].fillna("").apply(parse_genres)
    movies["overview"] = movies["overview"].fillna("").astype(str)
    movies["tagline"] = movies["tagline"].fillna("").astype(str)
    movies["rating"] = pd.to_numeric(movies["vote_average"], errors="coerce")
    movies["vote_count"] = pd.to_numeric(movies["vote_count"], errors="coerce").fillna(0)
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce").fillna(0)
    movies["year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
    movies["language"] = movies["original_language"].fillna("").replace("", pd.NA)
    movies["tags"] = (
        movies["overview"] + " " + movies["tagline"] + " "
        + movies["genres"].apply(" ".join)
    ).str.strip()
    movies = movies[movies["tags"].str.len() > 0].reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2, max_features=30_000
    )
    matrix = vectorizer.fit_transform(movies["tags"])
    artifact_columns = [
        "movie_id", "title", "overview", "tagline", "genres", "rating",
        "vote_count", "popularity", "year", "language", "poster_path", "tags",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in {
        "movies.pkl": movies[artifact_columns],
        "tfidf_matrix.pkl": matrix,
        "tfidf_vectorizer.pkl": vectorizer,
    }.items():
        with (output_dir / filename).open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Built artifacts for {len(movies):,} movies with {matrix.shape[1]:,} features in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "movies_metadata.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    build(args.source, args.output)
