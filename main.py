"""FastAPI backend for the CineMatch movie discovery app."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from recommender import MovieNotFoundError, MovieRecommender


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", BASE_DIR / "artifacts"))
DEFAULT_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501"


class MovieCard(BaseModel):
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    rating: float | None = None
    popularity: float | None = None
    language: str | None = None
    poster_url: str | None = None
    overview: str | None = None
    similarity: float | None = None
    reason: str | None = None


class MovieDetails(MovieCard):
    tagline: str | None = None
    tmdb_url: str | None = None


class DiscoverRequest(BaseModel):
    genres: list[str] = Field(default_factory=list, max_length=5)
    year_min: int | None = Field(default=None, ge=1870, le=2100)
    year_max: int | None = Field(default=None, ge=1870, le=2100)
    minimum_rating: float = Field(default=0, ge=0, le=10)
    languages: list[str] = Field(default_factory=list, max_length=5)
    popularity: Literal["balanced", "mainstream", "hidden_gems"] = "balanced"
    limit: int = Field(default=12, ge=1, le=30)


def get_recommender(request: Request) -> MovieRecommender:
    service = getattr(request.app.state, "recommender", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Recommendation model is not ready")
    return service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.recommender = MovieRecommender.from_directory(ARTIFACT_DIR)
    yield
    app.state.recommender = None


app = FastAPI(
    title="CineMatch API",
    description="Content-based recommendations and preference-driven movie discovery.",
    version="2.0.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    service = getattr(request.app.state, "recommender", None)
    return {
        "status": "ok" if service is not None else "starting",
        "model_loaded": service is not None,
        "movie_count": len(service.movies) if service is not None else 0,
    }


@app.get("/genres", response_model=list[str])
def genres(request: Request) -> list[str]:
    return get_recommender(request).genres


@app.get("/languages", response_model=dict[str, str])
def languages(request: Request) -> dict[str, str]:
    return get_recommender(request).languages


@app.get("/movies/search", response_model=list[MovieCard])
def search_movies(
    request: Request,
    query: Annotated[str, Query(min_length=2, max_length=100)],
    limit: Annotated[int, Query(ge=1, le=30)] = 12,
) -> list[dict]:
    return get_recommender(request).search(query, limit)


@app.get("/movies/{movie_id}", response_model=MovieDetails)
def movie_details(request: Request, movie_id: int) -> dict:
    try:
        return get_recommender(request).details(movie_id)
    except MovieNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/recommend/{movie_id}", response_model=list[MovieCard])
def recommendations(
    request: Request,
    movie_id: int,
    limit: Annotated[int, Query(ge=1, le=30)] = 12,
    diverse: bool = True,
) -> list[dict]:
    try:
        return get_recommender(request).recommend(movie_id, limit, diverse)
    except MovieNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/discover", response_model=list[MovieCard])
def discover(request: Request, preferences: DiscoverRequest) -> list[dict]:
    if (
        preferences.year_min is not None
        and preferences.year_max is not None
        and preferences.year_min > preferences.year_max
    ):
        raise HTTPException(status_code=422, detail="year_min must not exceed year_max")
    return get_recommender(request).discover(**preferences.model_dump())
