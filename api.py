"""
MovieMatch FastAPI Backend
Production-grade REST API for the MovieMatch recommendation system

Author: Sivarkamani
Purpose: Expose MovieMatch as an API that any frontend/service can consume

API Endpoints:
    GET  /              - Health check
    GET  /movies        - Get all movies
    POST /search        - Semantic search
    POST /similar       - Find similar movies
    GET  /trending      - Trending movies
    GET  /top-rated     - Top rated movies
    POST /ask           - AI Q&A about movies
    GET  /stats         - System statistics
    GET  /docs          - Auto-generated API docs (FastAPI built-in!)
"""

import time
import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core import (
    fetch_movies,
    fetch_trending,
    fetch_top_rated,
    build_vectorstore,
    semantic_search,
    ask_llm
)

# ============================================
# Pydantic Models (Request/Response schemas)
# ============================================

class SearchRequest(BaseModel):
    """Schema for semantic search request"""
    query: str = Field(
        ...,
        description="Natural language movie search query",
        example="mind-bending sci-fi about dreams"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (1-20)"
    )
    genre_filter: Optional[str] = Field(
        default=None,
        description="Filter by genre (e.g., 'Action', 'Drama')"
    )


class SimilarRequest(BaseModel):
    """Schema for similar movies request"""
    movie_title: str = Field(
        ...,
        description="Title of movie to find similar ones for",
        example="Inception"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of similar movies to return"
    )


class AskRequest(BaseModel):
    """Schema for AI Q&A request"""
    question: str = Field(
        ...,
        description="Question about a movie",
        example="What themes does Parasite explore?"
    )
    movie_context: str = Field(
        ...,
        description="Movie description/overview for context"
    )
    api_key: str = Field(
        ...,
        description="Gemini API key (get free key at aistudio.google.com/apikey)"
    )
    provider: str = Field(
        default="gemini",
        description="LLM provider: 'gemini' (free), 'openai' (paid), 'claude' (paid)"
    )


class MovieResponse(BaseModel):
    """Schema for movie data in responses"""
    title: str
    year: str
    genre: str
    rating: float
    votes: int
    poster: Optional[str]
    overview: str
    match_score: Optional[float] = None


class SearchResponse(BaseModel):
    """Schema for search response"""
    query: str
    total_results: int
    response_time_ms: float
    matches: list[MovieResponse]


class StatsResponse(BaseModel):
    """Schema for system statistics"""
    total_movies: int
    vector_dimensions: int
    embedding_model: str
    vector_db: str
    avg_response_time_ms: float
    api_version: str


# ============================================
# App State (shared across requests)
# ============================================
app_state = {
    "vectorstore": None,
    "movies": [],
    "response_times": [],
    "startup_time": None
}


# ============================================
# Lifespan (startup/shutdown events)
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize resources on startup
    This runs ONCE when the API starts
    """
    print("🚀 MovieMatch API Starting...")
    print("🎬 Loading movies from TMDB...")

    start = time.time()

    # Fetch movies from TMDB
    movies = fetch_movies(total_pages=5)
    app_state["movies"] = movies
    print(f"✅ Loaded {len(movies)} movies from TMDB")

    # Build vector store
    print("🧠 Building ChromaDB vector store...")
    vectorstore = build_vectorstore(movies)
    app_state["vectorstore"] = vectorstore
    print("✅ Vector store ready!")

    app_state["startup_time"] = time.time() - start
    print(f"🎉 API Ready in {app_state['startup_time']:.2f}s")

    yield  # API runs here

    # Cleanup on shutdown
    print("🛑 MovieMatch API Shutting down...")


# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title="🎬 MovieMatch API",
    description="""
    **AI-Powered Movie Recommendation System**
    
    Built with FastAPI + LangChain + ChromaDB + TMDB API
    
    ## Features
    - 🔍 **Semantic Search** — Find movies by vibe/mood/theme
    - 🎯 **Similar Movies** — "More like this" recommendations  
    - 🔥 **Trending** — This week's most popular movies
    - ⭐ **Top Rated** — All-time greatest movies
    - 🤖 **AI Q&A** — Chat with Gemini about any movie
    - 📊 **Analytics** — System performance stats
    
    ## Tech Stack
    - **FastAPI** — High-performance async API
    - **LangChain** — RAG pipeline orchestration
    - **ChromaDB** — Vector similarity search
    - **Sentence Transformers** — Free embeddings (no API key!)
    - **TMDB API** — 1M+ real movie database
    - **Gemini AI** — Free LLM for Q&A
    
    ## Author
    **Sivarkamani** | AI/ML Engineer
    """,
    version="2.0.0",
    lifespan=lifespan
)

# ============================================
# CORS Middleware
# Allows Streamlit and other frontends to call this API
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Helper: Track response times
# ============================================
def track_response_time(elapsed_ms: float):
    """Track response times for stats endpoint"""
    app_state["response_times"].append(elapsed_ms)
    # Keep only last 100 response times
    if len(app_state["response_times"]) > 100:
        app_state["response_times"].pop(0)


# ============================================
# API Endpoints
# ============================================

@app.get(
    "/",
    tags=["Health"],
    summary="Health Check"
)
async def health_check():
    """
    Check if the API is running and ready
    
    Returns system status and basic info
    """
    return {
        "status": "✅ MovieMatch API is running!",
        "version": "2.0.0",
        "movies_loaded": len(app_state["movies"]),
        "vector_store_ready": app_state["vectorstore"] is not None,
        "startup_time_seconds": round(app_state["startup_time"] or 0, 2),
        "message": "Visit /docs for interactive API documentation"
    }


@app.get(
    "/movies",
    tags=["Movies"],
    summary="Get All Movies",
    response_model=list[MovieResponse]
)
async def get_movies(
    limit: int = Query(default=20, ge=1, le=200, description="Number of movies to return"),
    genre: Optional[str] = Query(default=None, description="Filter by genre"),
    min_rating: float = Query(default=0.0, ge=0, le=10, description="Minimum rating filter"),
    sort_by: str = Query(default="rating", description="Sort by: rating, year, votes")
):
    """
    Get all movies in the database with optional filtering

    - **limit**: Max movies to return (1-200)
    - **genre**: Filter by genre name (e.g., 'Action', 'Drama')
    - **min_rating**: Only return movies above this rating
    - **sort_by**: Sort field (rating, year, votes)
    """
    movies = app_state["movies"]

    # Apply genre filter
    if genre:
        movies = [m for m in movies if genre.lower() in m["genre"].lower()]

    # Apply rating filter
    movies = [m for m in movies if m["rating"] >= min_rating]

    # Apply sorting
    if sort_by == "rating":
        movies = sorted(movies, key=lambda x: x["rating"], reverse=True)
    elif sort_by == "year":
        movies = sorted(movies, key=lambda x: x["year"], reverse=True)
    elif sort_by == "votes":
        movies = sorted(movies, key=lambda x: x["votes"], reverse=True)

    # Apply limit
    movies = movies[:limit]

    return [MovieResponse(**{**m, "match_score": None}) for m in movies]


@app.post(
    "/search",
    tags=["Search"],
    summary="Semantic Movie Search",
    response_model=SearchResponse
)
async def search_movies(request: SearchRequest):
    """
    Search movies using **semantic AI** — finds movies by meaning, not just keywords!

    ## Examples
    - "mind-bending sci-fi about dreams" → Inception, Interstellar
    - "feel-good movie about friendship" → The Shawshank Redemption
    - "intense psychological thriller" → Whiplash, Parasite

    ## How it works
    1. Your query is converted to a 384-dimensional embedding vector
    2. ChromaDB finds the most similar movie vectors using cosine similarity
    3. Top-k matches are returned with relevance scores
    """
    if not app_state["vectorstore"]:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Please wait for startup."
        )

    start = time.time()

    try:
        # Apply genre pre-filter if specified
        query = request.query
        if request.genre_filter:
            query = f"{request.genre_filter} movie: {query}"

        matches, response_time = semantic_search(
            query=query,
            vectorstore=app_state["vectorstore"],
            top_k=request.top_k
        )

        track_response_time(response_time)

        return SearchResponse(
            query=request.query,
            total_results=len(matches),
            response_time_ms=round(response_time, 2),
            matches=[MovieResponse(**m) for m in matches]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post(
    "/similar",
    tags=["Search"],
    summary="Find Similar Movies",
    response_model=SearchResponse
)
async def find_similar(request: SimilarRequest):
    """
    Find movies similar to one you already love!

    ## How it works
    Uses the movie's own description as a search query to find
    semantically similar movies in the database.

    ## Example
    - Input: "Inception"
    - Output: Interstellar, Eternal Sunshine, The Dark Knight
    """
    if not app_state["vectorstore"]:
        raise HTTPException(status_code=503, detail="Vector store not ready")

    # Find the target movie
    target = next(
        (m for m in app_state["movies"]
         if m["title"].lower() == request.movie_title.lower()),
        None
    )

    if not target:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{request.movie_title}' not found in database"
        )

    # Use movie description as search query (k+1 to exclude self)
    matches, response_time = semantic_search(
        query=target["overview"],
        vectorstore=app_state["vectorstore"],
        top_k=request.top_k + 1
    )

    # Filter out the original movie
    matches = [m for m in matches if m["title"].lower() != request.movie_title.lower()]
    matches = matches[:request.top_k]

    track_response_time(response_time)

    return SearchResponse(
        query=f"Similar to: {request.movie_title}",
        total_results=len(matches),
        response_time_ms=round(response_time, 2),
        matches=[MovieResponse(**m) for m in matches]
    )


@app.get(
    "/trending",
    tags=["Movies"],
    summary="Trending Movies This Week"
)
async def get_trending():
    """
    Get trending movies this week from TMDB

    Updated weekly with real trending data.
    """
    trending = fetch_trending()
    if not trending:
        raise HTTPException(status_code=503, detail="Could not fetch trending movies")
    return {
        "source": "TMDB Trending API",
        "period": "This Week",
        "total": len(trending),
        "movies": trending
    }


@app.get(
    "/top-rated",
    tags=["Movies"],
    summary="Top Rated Movies All Time"
)
async def get_top_rated():
    """
    Get highest rated movies of all time from TMDB

    Based on TMDB community ratings from millions of users.
    """
    top_rated = fetch_top_rated()
    if not top_rated:
        raise HTTPException(status_code=503, detail="Could not fetch top rated movies")
    return {
        "source": "TMDB Top Rated API",
        "total": len(top_rated),
        "movies": top_rated
    }


@app.post(
    "/ask",
    tags=["AI"],
    summary="Ask AI About a Movie"
)
async def ask_about_movie(request: AskRequest):
    """
    Ask an AI question about any movie using **Gemini AI**

    ## Recommended Provider
    Use **Gemini** (free) — 1,500 requests/day at no cost!
    Get your free key at: https://aistudio.google.com/apikey

    ## Cost Warning
    OpenAI and Claude require paid API access.
    Gemini is recommended for cost-effectiveness.

    ## Example Questions
    - "What themes does this movie explore?"
    - "Is this movie suitable for children?"
    - "What makes this movie unique?"
    """
    # Warn about non-Gemini providers
    if request.provider != "gemini":
        return JSONResponse(
            status_code=200,
            content={
                "provider": request.provider,
                "question": request.question,
                "answer": (
                    f"⚠️ {request.provider.capitalize()} is disabled for cost reasons. "
                    "Please use Gemini (free tier: 1,500 requests/day). "
                    "Get free key at: https://aistudio.google.com/apikey"
                ),
                "cost_warning": True
            }
        )

    try:
        answer = ask_llm(
            question=request.question,
            context=request.movie_context,
            api_key=request.api_key,
            provider=request.provider
        )

        return {
            "provider": request.provider,
            "question": request.question,
            "answer": answer,
            "cost_warning": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")


@app.get(
    "/stats",
    tags=["System"],
    summary="System Statistics",
    response_model=StatsResponse
)
async def get_stats():
    """
    Get system performance statistics

    Shows current database size, model info, and performance metrics.
    """
    avg_response = (
        sum(app_state["response_times"]) / len(app_state["response_times"])
        if app_state["response_times"] else 0
    )

    return StatsResponse(
        total_movies=len(app_state["movies"]),
        vector_dimensions=384,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_db="ChromaDB (cosine similarity)",
        avg_response_time_ms=round(avg_response, 2),
        api_version="2.0.0"
    )


@app.get(
    "/genres",
    tags=["Movies"],
    summary="Get Available Genres"
)
async def get_genres():
    """Get list of all available genres in the database"""
    genres = set()
    for movie in app_state["movies"]:
        for genre in movie["genre"].split("/"):
            if genre.strip():
                genres.add(genre.strip())

    return {
        "total_genres": len(genres),
        "genres": sorted(list(genres))
    }
