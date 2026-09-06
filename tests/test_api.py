import unittest

from fastapi.testclient import TestClient

from main import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def test_health_and_search(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["model_loaded"])

        response = self.client.get("/movies/search", params={"query": "Toy Story"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["title"], "Toy Story")
        self.assertEqual(response.json()[0]["year"], 1995)

    def test_recommendations_are_explained_and_unique(self) -> None:
        response = self.client.get("/recommend/862", params={"limit": 8})
        self.assertEqual(response.status_code, 200)
        recommendations = response.json()
        self.assertEqual(len(recommendations), 8)
        self.assertTrue(all(movie["reason"] for movie in recommendations))
        self.assertEqual(
            len({movie["title"].casefold() for movie in recommendations}), 8
        )

    def test_unknown_movie_returns_404(self) -> None:
        response = self.client.get("/movies/999999999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])

    def test_missing_poster_and_overview_are_safe(self) -> None:
        missing_poster = self.client.get("/movies/218473")
        self.assertEqual(missing_poster.status_code, 200)
        self.assertIsNone(missing_poster.json()["poster_url"])

        missing_overview = self.client.get("/movies/78802")
        self.assertEqual(missing_overview.status_code, 200)
        self.assertIsNone(missing_overview.json()["overview"])

    def test_discover_respects_preferences(self) -> None:
        response = self.client.post(
            "/discover",
            json={
                "genres": ["Animation"],
                "year_min": 1990,
                "year_max": 2010,
                "minimum_rating": 6.5,
                "languages": ["en"],
                "popularity": "balanced",
                "limit": 6,
            },
        )
        self.assertEqual(response.status_code, 200)
        movies = response.json()
        self.assertLessEqual(len(movies), 6)
        self.assertGreater(len(movies), 0)
        self.assertTrue(all("Animation" in movie["genres"] for movie in movies))
        self.assertTrue(all(1990 <= movie["year"] <= 2010 for movie in movies))
        self.assertTrue(all(movie["rating"] >= 6.5 for movie in movies))

    def test_discover_rejects_reversed_years(self) -> None:
        response = self.client.post(
            "/discover",
            json={"year_min": 2010, "year_max": 1990, "limit": 5},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
