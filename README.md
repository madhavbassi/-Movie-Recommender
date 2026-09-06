# CineMatch

CineMatch is a personal movie-discovery assistant that recommends films from their story descriptions, taglines, and genres. It combines an explainable content-based model with practical preference filters, a FastAPI backend, and a Streamlit interface.

It is Version 2 of the original **Movie Recommender** student project, repositioned from a generic Netflix-style interface into a focused discovery assistant.

Unlike a basic title-to-title recommender, CineMatch supports duplicate titles through stable movie IDs, explains each recommendation, offers mainstream and hidden-gem discovery modes, reduces repetitive results, and includes a lightweight session watchlist.

## Features

- Searchable movie selection with release years for duplicate titles
- TF-IDF and cosine-similarity recommendations
- Truthful recommendation reasons based on shared genres and text features
- Filters for genre, release year, rating, language, and popularity style
- Simple result diversity to avoid repeated titles and one-genre domination
- Posters, overview, rating, genres, release year, and TMDB links where available
- Session-only watchlist with a CSV download, no login, and no database
- Graceful empty states, missing metadata handling, and backend timeout messages
- Validated FastAPI requests, narrow configurable CORS, and a health endpoint

## Recommendation approach

The build script combines each movie's overview, tagline, and genres into a text profile. `TfidfVectorizer` represents those profiles as weighted word and phrase features. Recommendations are ranked by cosine similarity to the selected movie. The API then applies a small diversity rule and creates an explanation using the genres or TF-IDF terms actually shared by the two movies.

Preference discovery is a transparent ranking rather than a learned user model. It filters the dataset and combines rating, popularity rank, and vote-count rank differently for balanced, mainstream, and hidden-gem modes.

## Tech stack

- Python 3.12
- pandas, NumPy, SciPy, scikit-learn
- FastAPI and Uvicorn
- Streamlit and Requests
- Movie metadata and poster paths from the TMDB Movies dataset

## Architecture

```text
Streamlit UI (app.py)
        |
        | JSON over HTTP
        v
FastAPI API (main.py)
        |
        v
MovieRecommender (recommender.py)
        |
        v
Local dataframe + TF-IDF artifacts
```

The API and frontend remain separate because that demonstrates API design while keeping the recommendation code independently testable. Poster images are loaded from TMDB's public image CDN using paths already present in the dataset; a TMDB API key is not required.

## Project structure

```text
.
├── app.py                         # Streamlit frontend
├── main.py                        # FastAPI routes and validation
├── recommender.py                 # Search, recommendations, filters, explanations
├── build_artifacts.py             # Reproducible model build
├── artifacts/
│   ├── movies.pkl
│   ├── tfidf_matrix.pkl
│   └── tfidf_vectorizer.pkl
├── data/movies_metadata.csv       # Source dataset
├── notebooks/project.ipynb        # Original exploratory notebook
├── tests/                         # Core and API tests
├── requirements.txt
├── dev-requirements.txt
├── render.yaml
└── .env.example
```

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to change the defaults. Do not commit `.env`.

### Generate model artifacts

The generated artifacts are included so deployments can start without training. Rebuild them whenever the source data or feature logic changes:

```bash
python build_artifacts.py
```

Pickle files can execute Python code while loading. Only load the checked-in artifacts or files you generated yourself with this script. Serialized scikit-learn objects should be rebuilt when changing the pinned scikit-learn version.

### Run the backend

```bash
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for interactive API documentation.

### Run the frontend

In a second terminal with the same environment:

```bash
streamlit run app.py
```

The frontend opens at `http://localhost:8501` and uses `http://localhost:8000` by default.

## Environment variables

| Variable | Service | Purpose | Default |
|---|---|---|---|
| `MOVIE_API_BASE` | Streamlit | Public URL of the FastAPI service | `http://localhost:8000` |
| `ALLOWED_ORIGINS` | FastAPI | Comma-separated permitted frontend origins | Local Streamlit origins |
| `API_TIMEOUT_SECONDS` | Streamlit | Backend request timeout | `35` |
| `ARTIFACT_DIR` | FastAPI | Optional alternate artifact directory | `./artifacts` |

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service and model readiness |
| `GET` | `/genres` | Available discovery genres |
| `GET` | `/languages` | Available language codes and names |
| `GET` | `/movies/search?query=...` | Search local titles |
| `GET` | `/movies/{movie_id}` | Movie metadata |
| `GET` | `/recommend/{movie_id}` | Explained content recommendations |
| `POST` | `/discover` | Preference-filtered discovery |

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover artifact loading, health and search endpoints, recommendation explanations and diversity, unknown IDs, preference filters, invalid year ranges, and the standalone recommendation class.

## Deployment

### FastAPI on Render

1. Push this repository to GitHub.
2. In Render, create a Blueprint from the repository; `render.yaml` supplies the build, start, Python version, and health-check settings.
3. Set `ALLOWED_ORIGINS` to the exact Streamlit Community Cloud URL, without a trailing slash.
4. Deploy and verify `/health` and `/docs` on the public Render URL.

### Streamlit Community Cloud

1. Create an app from the same repository and choose `app.py` as the entry point.
2. In app secrets, add the deployed backend URL:

   ```toml
   MOVIE_API_BASE = "https://your-render-service.onrender.com"
   ```

   The frontend reads this value from either Streamlit secrets or a normal environment variable.
3. Confirm the Streamlit public URL exactly matches `ALLOWED_ORIGINS` on Render.
4. Test search, details, recommendations, and discovery from the public app.

Free hosting can take time to wake after inactivity. The frontend uses a 35-second timeout and shows a retry message instead of treating a cold start as an application failure.

## Limitations

- The dataset ends in 2017, so this is not a current-release catalogue.
- Ratings and popularity are historical dataset values, not live TMDB values.
- Content similarity does not learn from individual user behavior.
- The watchlist is temporary Streamlit session state and is not shared across devices.
- Pickled artifacts are tied to compatible Python library versions and must come from a trusted source.

## Future improvements

- Refresh metadata through a documented data update pipeline.
- Add evaluation metrics and a small human relevance test set.
- Persist watchlists only if authentication becomes a real requirement.
- Add optional live TMDB enrichment without making it a startup dependency.

## Screenshots

Add screenshots of the search, recommendation explanations, and discovery filters after the deployed UI is verified.

## Resume presentation

**Project title:** CineMatch — Explainable Movie Discovery Assistant

**GitHub description:** Explainable content-based movie discovery with TF-IDF, preference filters, FastAPI, and Streamlit.

**Resume bullets:**

- Built a content-based movie discovery system over 45,000 titles using TF-IDF and cosine similarity, with explanations derived from shared genres and text features.
- Designed validated FastAPI endpoints for search, recommendations, and preference filtering, then integrated them with a responsive Streamlit interface.
- Added reproducible model generation, duplicate-title handling, diversity rules, automated API tests, and deployment configuration for Render and Streamlit Community Cloud.

## Data acknowledgement

The project uses the public TMDB movie metadata dataset for educational purposes. This product uses the TMDB API image service but is not endorsed or certified by TMDB.
