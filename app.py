"""Streamlit frontend for CineMatch."""

import csv
from html import escape
from io import StringIO
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st


st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")


def setting(name: str, default: str) -> str:
    environment_value = os.getenv(name)
    if environment_value:
        return environment_value
    local_secrets = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    user_secrets = Path.home() / ".streamlit" / "secrets.toml"
    if local_secrets.exists() or user_secrets.exists():
        return str(st.secrets.get(name, default))
    return default


API_BASE = setting("MOVIE_API_BASE", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(setting("API_TIMEOUT_SECONDS", "35"))

st.markdown(
    """
    <style>
    .block-container {max-width: 1240px; padding-top: 1.25rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.16);}
    [data-testid="stSidebar"] .block-container {padding-top: 1.6rem;}
    .hero {
        padding: 2rem 2.2rem; border-radius: 22px; margin-bottom: 1.6rem;
        color: white; background:
        radial-gradient(circle at 85% 15%, rgba(255,255,255,.18), transparent 24%),
        linear-gradient(120deg, #25164f 0%, #6f2c76 55%, #b74765 100%);
        box-shadow: 0 14px 34px rgba(44, 21, 83, .2);
    }
    .hero-kicker {font-size: .78rem; font-weight: 700; letter-spacing: .14em; opacity: .78;}
    .hero h1 {font-size: 2.45rem; line-height: 1.05; margin: .5rem 0 .7rem;}
    .hero p {font-size: 1.02rem; max-width: 670px; margin: 0; opacity: .88;}
    .section-copy {color: #81889a; margin-top: -.5rem; margin-bottom: 1rem;}
    .movie-title {font-size: 1.02rem; font-weight: 700; line-height: 1.25; min-height: 2.55rem;}
    .movie-meta {color: #848b9c; font-size: .82rem; min-height: 2.3rem; margin-top: .25rem;}
    .movie-copy {font-size: .86rem; line-height: 1.42; min-height: 3.7rem; margin: .4rem 0 .6rem;}
    .poster-shell {position: relative; aspect-ratio: 2/3; margin-bottom: .7rem;}
    .poster-shell img {
        position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;
        border-radius: 13px;
    }
    .poster-placeholder {
        position: absolute; inset: 0; display: flex; flex-direction: column;
        align-items: center; justify-content: center; text-align: center; padding: 1rem;
        border-radius: 13px; color: #d9d4e8;
        background: linear-gradient(145deg, #242034, #3b3154);
    }
    .poster-placeholder span {font-size: 2rem; margin-bottom: .4rem;}
    .status-pill {font-size: .82rem; color: #5cbf8b; margin-bottom: .6rem;}
    .detail-facts {color: #8b91a0; margin: .4rem 0 1rem;}
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px; border-color: rgba(128,128,128,.18);
        box-shadow: 0 5px 18px rgba(20,20,35,.05);
    }
    div.stButton > button, div.stDownloadButton > button {border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_STATE = {
    "watchlist": {},
    "selected_movie_id": None,
    "search_query": "",
    "discovery_results": None,
    "discovery_summary": "",
}
for state_key, default_value in DEFAULT_STATE.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str, params: dict[str, Any] | None = None):
    try:
        response = requests.get(
            f"{API_BASE}{path}", params=params, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json(), None
    except requests.Timeout:
        return None, (
            "The backend is taking longer than expected. A free deployment may be "
            "waking up; please try again shortly."
        )
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                detail = exc.response.text[:160]
        return None, detail or "The movie service is currently unavailable."


def api_post(path: str, payload: dict[str, Any]):
    try:
        response = requests.post(
            f"{API_BASE}{path}", json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json(), None
    except requests.Timeout:
        return None, "The backend is taking longer than expected. Please try again shortly."
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                detail = exc.response.text[:160]
        return None, detail or "The movie service is currently unavailable."


def open_movie(movie_id: int) -> None:
    st.session_state.selected_movie_id = movie_id


def set_search(title: str) -> None:
    st.session_state.search_query = title


def toggle_watchlist(movie: dict) -> None:
    movie_id = movie["movie_id"]
    if movie_id in st.session_state.watchlist:
        del st.session_state.watchlist[movie_id]
    else:
        st.session_state.watchlist[movie_id] = movie


def movie_copy(movie: dict) -> str:
    if movie.get("reason"):
        return escape(movie["reason"])
    overview = (movie.get("overview") or "Overview unavailable.").strip()
    return escape(overview[:125] + ("…" if len(overview) > 125 else ""))


def render_poster(movie: dict) -> None:
    title = escape(movie.get("title", "Movie"))
    poster = movie.get("poster_url")
    image = ""
    if poster:
        safe_url = escape(poster, quote=True)
        image = f'<img src="{safe_url}" alt="Poster for {title}" onerror="this.style.display=\'none\'">'
    st.markdown(
        f"""
        <div class="poster-shell">
          <div class="poster-placeholder"><span>🎞️</span>{title}<br><small>Poster unavailable</small></div>
          {image}
        </div>
        """,
        unsafe_allow_html=True,
    )


def movie_grid(
    movies: list[dict],
    key_prefix: str,
    columns: int = 3,
    empty_message: str = "No movies matched. Try widening your choices.",
) -> None:
    if not movies:
        st.info(empty_message)
        return

    for start in range(0, len(movies), columns):
        for column, movie in zip(st.columns(columns), movies[start : start + columns]):
            with column:
                with st.container(border=True):
                    render_poster(movie)

                    st.markdown(
                        f"<div class='movie-title'>{escape(movie['title'])}</div>",
                        unsafe_allow_html=True,
                    )
                    meta = " · ".join(
                        value
                        for value in [
                            str(movie.get("year") or "Year unknown"),
                            ", ".join(movie.get("genres", [])[:2]),
                            f"★ {movie['rating']:.1f}" if movie.get("rating") else "",
                        ]
                        if value
                    )
                    st.markdown(
                        f"<div class='movie-meta'>{escape(meta)}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='movie-copy'>{movie_copy(movie)}</div>",
                        unsafe_allow_html=True,
                    )

                    left, right = st.columns(2)
                    left.button(
                        "View details",
                        key=f"{key_prefix}_open_{movie['movie_id']}",
                        on_click=open_movie,
                        args=(movie["movie_id"],),
                        use_container_width=True,
                    )
                    saved = movie["movie_id"] in st.session_state.watchlist
                    right.button(
                        "✓ Saved" if saved else "+ Save",
                        key=f"{key_prefix}_watch_{movie['movie_id']}",
                        on_click=toggle_watchlist,
                        args=(movie,),
                        type="primary" if saved else "secondary",
                        use_container_width=True,
                    )


def watchlist_csv() -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["title", "year", "genres", "rating", "tmdb_url"])
    for movie in st.session_state.watchlist.values():
        writer.writerow(
            [
                movie.get("title", ""),
                movie.get("year", ""),
                ", ".join(movie.get("genres", [])),
                movie.get("rating", ""),
                f"https://www.themoviedb.org/movie/{movie['movie_id']}",
            ]
        )
    return output.getvalue()


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">PERSONAL MOVIE DISCOVERY</div>
      <h1>Find a film worth your time.</h1>
      <p>Start with a movie you love or shape a shortlist around your mood. Every
      recommendation includes a simple, honest reason.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    page = st.radio(
        "Navigation",
        ["🔎 Find similar", "✨ Discover", "♡ Watchlist"],
        label_visibility="collapsed",
    )
    st.divider()
    health, _ = api_get("/health")
    if health and health.get("model_loaded"):
        st.markdown(
            f"<div class='status-pill'>● Library ready · {health['movie_count']:,} titles</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Movie service is starting or unavailable.")
    saved_count = len(st.session_state.watchlist)
    st.metric("Saved movies", saved_count)
    st.caption("Your watchlist lasts for this browser session.")


if st.session_state.selected_movie_id is not None:
    movie, error = api_get(f"/movies/{st.session_state.selected_movie_id}")
    if error:
        st.error(error)
        if st.button("← Go back"):
            st.session_state.selected_movie_id = None
            st.rerun()
    else:
        top_left, top_right = st.columns([4, 1])
        with top_left:
            if st.button("← Back to results"):
                st.session_state.selected_movie_id = None
                st.rerun()
        with top_right:
            saved = movie["movie_id"] in st.session_state.watchlist
            st.button(
                "✓ In watchlist" if saved else "+ Add to watchlist",
                on_click=toggle_watchlist,
                args=(movie,),
                type="primary",
                use_container_width=True,
            )

        poster_col, detail_col = st.columns([1, 2.3], gap="large")
        with poster_col:
            render_poster(movie)
        with detail_col:
            st.header(movie["title"])
            facts = [str(movie.get("year") or "Year unknown")]
            if movie.get("genres"):
                facts.append(", ".join(movie["genres"]))
            if movie.get("rating"):
                facts.append(f"★ {movie['rating']:.1f}/10")
            st.markdown(
                f"<div class='detail-facts'>{escape(' · '.join(facts))}</div>",
                unsafe_allow_html=True,
            )
            if movie.get("tagline"):
                st.markdown(f"### *{escape(movie['tagline'])}*")
            st.write(movie.get("overview") or "No overview is available.")
            if movie.get("tmdb_url"):
                st.link_button("Open on TMDB ↗", movie["tmdb_url"])

        st.divider()
        st.subheader("Because you chose this")
        st.markdown(
            "<div class='section-copy'>Ranked by story, theme, and genre similarity—not popularity alone.</div>",
            unsafe_allow_html=True,
        )
        recommendations, error = api_get(
            f"/recommend/{movie['movie_id']}", {"limit": 9, "diverse": True}
        )
        if error:
            st.error(error)
        else:
            movie_grid(recommendations, "recommendations")
    st.stop()


if page == "🔎 Find similar":
    st.subheader("Start with a movie you already like")
    st.markdown(
        "<div class='section-copy'>Search the library, choose the right release year, then explore similar stories.</div>",
        unsafe_allow_html=True,
    )
    st.text_input(
        "Movie title",
        key="search_query",
        placeholder="Search for Toy Story, The Godfather, Avatar…",
    )

    if not st.session_state.search_query:
        st.caption("Quick starts")
        quick_columns = st.columns(4)
        for column, title in zip(
            quick_columns, ["Toy Story", "The Godfather", "Spirited Away", "Inception"]
        ):
            column.button(
                title,
                on_click=set_search,
                args=(title,),
                use_container_width=True,
            )

    query = st.session_state.search_query.strip()
    if len(query) >= 2:
        with st.spinner("Searching the library…"):
            results, error = api_get("/movies/search", {"query": query, "limit": 12})
        if error:
            st.error(error)
        else:
            result_label = "result" if len(results) == 1 else "results"
            st.markdown(f"#### {len(results)} {result_label} for “{escape(query)}”")
            st.caption("Release years help distinguish movies with the same title.")
            movie_grid(
                results,
                "search",
                empty_message="No title matched that search. Check the spelling or try fewer words.",
            )
    elif query:
        st.caption("Enter at least two characters.")


elif page == "✨ Discover":
    st.subheader("Build a shortlist around your mood")
    st.markdown(
        "<div class='section-copy'>Choose only what matters to you. Leaving genres or languages blank keeps the search broad.</div>",
        unsafe_allow_html=True,
    )
    available_genres, genre_error = api_get("/genres")
    available_languages, language_error = api_get("/languages")
    if genre_error or language_error:
        st.error(genre_error or language_error)
    else:
        with st.form("discover_form"):
            first, second = st.columns(2, gap="large")
            with first:
                genres = st.multiselect(
                    "Genres", available_genres, max_selections=5,
                    help="Pick up to five. A movie may match any selected genre.",
                )
                year_min, year_max = st.slider(
                    "Release year", 1900, 2017, (1980, 2017)
                )
                minimum_rating = st.slider(
                    "Minimum rating", 0.0, 10.0, 6.0, 0.5,
                    help="Ratings come from the historical dataset.",
                )
            with second:
                languages = st.multiselect(
                    "Languages",
                    options=list(available_languages),
                    format_func=lambda code: available_languages[code],
                    max_selections=5,
                )
                popularity = st.radio(
                    "Discovery style",
                    ["balanced", "mainstream", "hidden_gems"],
                    format_func=lambda value: value.replace("_", " ").title(),
                    help="Balanced mixes quality and reach. Hidden gems favors highly rated, lower-profile titles.",
                )
                limit = st.slider("Number of results", 3, 18, 9)
            submitted = st.form_submit_button(
                "Find movies", type="primary", use_container_width=True
            )

        if submitted:
            payload = {
                "genres": genres,
                "year_min": year_min,
                "year_max": year_max,
                "minimum_rating": minimum_rating,
                "languages": languages,
                "popularity": popularity,
                "limit": limit,
            }
            with st.spinner("Building your shortlist…"):
                results, error = api_post("/discover", payload)
            if error:
                st.error(error)
            else:
                st.session_state.discovery_results = results
                style = popularity.replace("_", " ")
                selected_genres = ", ".join(genres) if genres else "all genres"
                st.session_state.discovery_summary = (
                    f"{style.title()} picks · {selected_genres} · {year_min}–{year_max}"
                )

        if st.session_state.discovery_results is not None:
            st.divider()
            st.subheader("Your shortlist")
            st.caption(st.session_state.discovery_summary)
            movie_grid(st.session_state.discovery_results, "discover")


else:
    heading, action = st.columns([3, 1])
    with heading:
        st.subheader("Your watchlist")
        st.markdown(
            "<div class='section-copy'>A lightweight shortlist saved for this browser session.</div>",
            unsafe_allow_html=True,
        )
    if st.session_state.watchlist:
        with action:
            st.download_button(
                "Download list",
                watchlist_csv(),
                file_name="cinematch-watchlist.csv",
                mime="text/csv",
                use_container_width=True,
            )
        movie_grid(list(st.session_state.watchlist.values()), "watchlist")
    else:
        st.info("Your watchlist is empty. Save a movie from search, discovery, or recommendations.")
