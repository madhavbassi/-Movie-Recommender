import unittest

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from recommender import MovieNotFoundError, MovieRecommender


def small_recommender() -> MovieRecommender:
    movies = pd.DataFrame(
        [
            {"movie_id": 1, "title": "Space One", "overview": "space crew adventure", "tagline": "", "genres": ["Science Fiction"], "rating": 8.0, "vote_count": 500, "popularity": 9.0, "year": 2001, "language": "en", "poster_path": None, "tags": "space crew adventure science fiction"},
            {"movie_id": 2, "title": "Space Two", "overview": "space crew mission", "tagline": "", "genres": ["Science Fiction"], "rating": 7.5, "vote_count": 300, "popularity": 7.0, "year": 2003, "language": "en", "poster_path": None, "tags": "space crew mission science fiction"},
            {"movie_id": 3, "title": "Quiet Drama", "overview": "family relationship", "tagline": "", "genres": ["Drama"], "rating": 7.2, "vote_count": 100, "popularity": 3.0, "year": 2005, "language": "fr", "poster_path": None, "tags": "family relationship drama"},
        ]
    )
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(movies["tags"])
    return MovieRecommender(movies, matrix, vectorizer)


class RecommenderTests(unittest.TestCase):
    def test_core_recommendation_and_reason(self) -> None:
        recommendation = small_recommender().recommend(1, limit=1)[0]
        self.assertEqual(recommendation["movie_id"], 2)
        self.assertIn("science fiction", recommendation["reason"].lower())

    def test_core_missing_movie(self) -> None:
        with self.assertRaises(MovieNotFoundError):
            small_recommender().details(99)


if __name__ == "__main__":
    unittest.main()
