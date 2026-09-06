"""Core content-based recommendation and preference filtering logic."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "hi": "Hindi",
    "zh": "Chinese", "ru": "Russian", "pt": "Portuguese", "ar": "Arabic",
    "tr": "Turkish", "sv": "Swedish", "da": "Danish", "nl": "Dutch",
}


class MovieNotFoundError(LookupError):
    pass


class MovieRecommender:
    def __init__(
        self,
        movies: pd.DataFrame,
        matrix: sparse.spmatrix,
        vectorizer: TfidfVectorizer,
    ) -> None:
        if len(movies) != matrix.shape[0]:
            raise ValueError("Movie data and TF-IDF matrix have different row counts")
        self.movies = movies.reset_index(drop=True)
        self.matrix = matrix.tocsr()
        self.vectorizer = vectorizer
        self._row_by_id = {
            int(movie_id): row for row, movie_id in enumerate(self.movies["movie_id"])
        }
        genres = {g for value in self.movies["genres"] for g in self._as_list(value)}
        self.genres = sorted(genres)
        used_languages = sorted(self.movies["language"].dropna().astype(str).unique())
        self.languages = {code: LANGUAGE_NAMES.get(code, code.upper()) for code in used_languages}

    @classmethod
    def from_directory(cls, directory: Path) -> "MovieRecommender":
        required = {
            "movies": directory / "movies.pkl",
            "matrix": directory / "tfidf_matrix.pkl",
            "vectorizer": directory / "tfidf_vectorizer.pkl",
        }
        missing = [path.name for path in required.values() if not path.exists()]
        if missing:
            raise RuntimeError(
                f"Missing model artifacts: {', '.join(missing)}. Run python build_artifacts.py."
            )
        with required["movies"].open("rb") as handle:
            movies = pickle.load(handle)
        with required["matrix"].open("rb") as handle:
            matrix = pickle.load(handle)
        with required["vectorizer"].open("rb") as handle:
            vectorizer = pickle.load(handle)
        return cls(movies, matrix, vectorizer)

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [item.strip() for item in value.split("|") if item.strip()]
        return []

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if pd.isna(value) else round(float(value), 2)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if pd.isna(value) else int(value)

    def _card(self, row: pd.Series, **extra: Any) -> dict[str, Any]:
        poster_path = row.get("poster_path")
        has_poster = pd.notna(poster_path) and str(poster_path).startswith("/")
        overview = str(row.get("overview") or "").strip() or None
        return {
            "movie_id": int(row["movie_id"]),
            "title": str(row["title"]),
            "year": self._optional_int(row.get("year")),
            "genres": self._as_list(row.get("genres")),
            "rating": self._optional_float(row.get("rating")),
            "popularity": self._optional_float(row.get("popularity")),
            "language": str(row.get("language")) if pd.notna(row.get("language")) else None,
            "poster_url": f"{TMDB_IMAGE_BASE}{poster_path}" if has_poster else None,
            "overview": overview,
            **extra,
        }

    def search(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        query = query.strip().casefold()
        titles = self.movies["title"].astype(str)
        matched = self.movies[titles.str.casefold().str.contains(query, regex=False)].copy()
        if matched.empty:
            return []
        matched["_starts"] = titles.loc[matched.index].str.casefold().str.startswith(query)
        matched["_exact"] = titles.loc[matched.index].str.casefold().eq(query)
        matched = matched.sort_values(
            ["_exact", "_starts", "popularity", "year"],
            ascending=[False, False, False, False],
            na_position="last",
        )
        return [self._card(row) for _, row in matched.head(limit).iterrows()]

    def details(self, movie_id: int) -> dict[str, Any]:
        row_number = self._row_by_id.get(movie_id)
        if row_number is None:
            raise MovieNotFoundError(f"Movie ID {movie_id} was not found")
        row = self.movies.iloc[row_number]
        tagline = str(row.get("tagline") or "").strip() or None
        return {
            **self._card(row),
            "tagline": tagline,
            "tmdb_url": f"https://www.themoviedb.org/movie/{movie_id}",
        }

    def _reason(self, source_row: pd.Series, candidate_row: pd.Series, terms: list[str]) -> str:
        source_genres = set(self._as_list(source_row.get("genres")))
        candidate_genres = set(self._as_list(candidate_row.get("genres")))
        shared_genres = sorted(source_genres & candidate_genres)
        if shared_genres:
            label = ", ".join(shared_genres[:2])
            return f"Similar {label.lower()} themes and story content."
        if terms:
            return f"Related story themes: {', '.join(terms[:3])}."
        return "Similar overall story description and themes."

    def _shared_terms(self, source_index: int, candidate_index: int) -> list[str]:
        overlap = self.matrix[source_index].multiply(self.matrix[candidate_index])
        if overlap.nnz == 0:
            return []
        best = overlap.indices[np.argsort(overlap.data)[::-1]][:5]
        return [str(term) for term in self.vectorizer.get_feature_names_out()[best]]

    def recommend(self, movie_id: int, limit: int = 12, diverse: bool = True) -> list[dict[str, Any]]:
        source_index = self._row_by_id.get(movie_id)
        if source_index is None:
            raise MovieNotFoundError(f"Movie ID {movie_id} was not found")
        scores = (self.matrix @ self.matrix[source_index].T).toarray().ravel()
        candidate_indices = np.argsort(-scores)
        source_row = self.movies.iloc[source_index]
        selected: list[dict[str, Any]] = []
        title_counts: dict[str, int] = {}
        lead_genre_counts: dict[str, int] = {}
        for candidate_index in candidate_indices:
            if int(candidate_index) == source_index or scores[candidate_index] <= 0:
                continue
            row = self.movies.iloc[int(candidate_index)]
            normalized_title = str(row["title"]).casefold()
            genres = self._as_list(row.get("genres"))
            lead_genre = genres[0] if genres else ""
            if diverse and (title_counts.get(normalized_title, 0) >= 1 or lead_genre_counts.get(lead_genre, 0) >= 4):
                continue
            terms = self._shared_terms(source_index, int(candidate_index))
            selected.append(
                self._card(
                    row,
                    similarity=round(float(scores[candidate_index]), 4),
                    reason=self._reason(source_row, row, terms),
                )
            )
            title_counts[normalized_title] = title_counts.get(normalized_title, 0) + 1
            if lead_genre:
                lead_genre_counts[lead_genre] = lead_genre_counts.get(lead_genre, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    def discover(
        self,
        genres: list[str],
        year_min: int | None,
        year_max: int | None,
        minimum_rating: float,
        languages: list[str],
        popularity: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        choices = self.movies.copy()
        if genres:
            wanted = set(genres)
            choices = choices[choices["genres"].apply(lambda value: bool(wanted & set(self._as_list(value))))]
        if year_min is not None:
            choices = choices[choices["year"] >= year_min]
        if year_max is not None:
            choices = choices[choices["year"] <= year_max]
        choices = choices[choices["rating"].fillna(0) >= minimum_rating]
        if languages:
            choices = choices[choices["language"].isin(languages)]
        if choices.empty:
            return []

        rating = choices["rating"].fillna(0) / 10
        popularity_rank = choices["popularity"].fillna(0).rank(pct=True)
        votes_rank = choices["vote_count"].fillna(0).rank(pct=True)
        if popularity == "mainstream":
            choices["_score"] = 0.25 * rating + 0.45 * popularity_rank + 0.30 * votes_rank
        elif popularity == "hidden_gems":
            credible = choices["vote_count"].fillna(0) >= 20
            choices = choices[credible]
            rating = choices["rating"].fillna(0) / 10
            popularity_rank = choices["popularity"].fillna(0).rank(pct=True)
            choices["_score"] = 0.75 * rating + 0.25 * (1 - popularity_rank)
        else:
            choices["_score"] = 0.50 * rating + 0.25 * popularity_rank + 0.25 * votes_rank

        choices = choices.sort_values("_score", ascending=False)
        results: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for _, row in choices.iterrows():
            title_key = str(row["title"]).casefold()
            if title_key in seen_titles:
                continue
            matched_genres = sorted(set(genres) & set(self._as_list(row.get("genres"))))
            reason_parts = []
            if matched_genres:
                reason_parts.append(f"matches {', '.join(matched_genres[:2])}")
            if row.get("rating", 0) >= 7:
                reason_parts.append(f"has a rating of {float(row['rating']):.1f}/10")
            if popularity == "hidden_gems":
                reason_parts.append("is a lower-profile pick")
            reason = "Recommended because it " + " and ".join(reason_parts) + "." if reason_parts else "Strong match for your selected filters."
            results.append(self._card(row, reason=reason))
            seen_titles.add(title_key)
            if len(results) >= limit:
                break
        return results
