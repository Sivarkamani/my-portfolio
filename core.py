"""
MovieMatch Core Module
Shared logic between Streamlit UI and FastAPI backend

Author: Sivarkamani
Purpose: Separation of concerns — business logic lives here
         Both app.py (Streamlit) and api.py (FastAPI) use this
"""

import os
import time
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# ============================================
# Constants
# ============================================
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"

# ============================================
# TMDB Functions
# ============================================

def fetch_genres() -> dict:
    """
    Fetch genre ID to name mapping from TMDB
    Returns: {genre_id: genre_name}
    """
    if not TMDB_API_KEY:
        return {}
    try:
        url = f"{TMDB_BASE_URL}/genre/movie/list"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            genres = response.json().get("genres", [])
            return {g["id"]: g["name"] for g in genres}
    except Exception:
        pass
    return {}


def fetch_movies(total_pages: int = 5) -> list:
    """
    Fetch popular movies from TMDB API

    Args:
        total_pages: Number of pages to fetch (20 movies per page)

    Returns:
        List of movie dictionaries with full metadata
    """
    if not TMDB_API_KEY:
        raise ValueError("TMDB_API_KEY not found in environment variables")

    movies = []
    genre_map = fetch_genres()

    for page in range(1, total_pages + 1):
        try:
            url = f"{TMDB_BASE_URL}/movie/popular"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "page": page
            }
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                for movie in response.json().get("results", []):
                    genre_names = [
                        genre_map.get(gid, "Unknown")
                        for gid in movie.get("genre_ids", [])[:3]
                    ]
                    movies.append({
                        "id": movie.get("id"),
                        "title": movie.get("title", "Unknown"),
                        "year": movie.get("release_date", "")[:4] or "N/A",
                        "rating": round(movie.get("vote_average", 0), 1),
                        "votes": movie.get("vote_count", 0),
                        "overview": movie.get("overview", "No description available"),
                        "genre": "/".join(genre_names) if genre_names else "Unknown",
                        "poster": f"{TMDB_IMAGE_BASE}{movie.get('poster_path')}" if movie.get("poster_path") else None,
                        "popularity": movie.get("popularity", 0),
                        "language": movie.get("original_language", "en")
                    })
            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"Error fetching page {page}: {str(e)}")
            continue

    return movies


def fetch_trending() -> list:
    """Fetch trending movies this week from TMDB"""
    if not TMDB_API_KEY:
        return []
    try:
        url = f"{TMDB_BASE_URL}/trending/movie/week"
        params = {"api_key": TMDB_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            genre_map = fetch_genres()
            results = []
            for movie in response.json().get("results", [])[:10]:
                genre_names = [
                    genre_map.get(gid, "")
                    for gid in movie.get("genre_ids", [])[:2]
                ]
                results.append({
                    "title": movie.get("title", ""),
                    "rating": round(movie.get("vote_average", 0), 1),
                    "year": movie.get("release_date", "")[:4],
                    "poster": f"{TMDB_IMAGE_BASE}{movie.get('poster_path')}" if movie.get("poster_path") else None,
                    "genre": "/".join(genre_names),
                    "overview": movie.get("overview", "")
                })
            return results
    except Exception:
        pass
    return []


def fetch_top_rated() -> list:
    """Fetch top rated movies from TMDB"""
    if not TMDB_API_KEY:
        return []
    try:
        url = f"{TMDB_BASE_URL}/movie/top_rated"
        params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": 1}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            genre_map = fetch_genres()
            results = []
            for movie in response.json().get("results", [])[:10]:
                genre_names = [
                    genre_map.get(gid, "")
                    for gid in movie.get("genre_ids", [])[:2]
                ]
                results.append({
                    "title": movie.get("title", ""),
                    "rating": round(movie.get("vote_average", 0), 1),
                    "year": movie.get("release_date", "")[:4],
                    "poster": f"{TMDB_IMAGE_BASE}{movie.get('poster_path')}" if movie.get("poster_path") else None,
                    "genre": "/".join(genre_names),
                    "overview": movie.get("overview", "")
                })
            return results
    except Exception:
        pass
    return []


# ============================================
# Vector Store Functions
# ============================================

def build_vectorstore(movies: list) -> Chroma:
    """
    Build ChromaDB vector store from movie list

    Args:
        movies: List of movie dictionaries from TMDB

    Returns:
        ChromaDB vector store ready for similarity search
    """
    docs = []
    for movie in movies:
        full_text = f"""
        Movie: {movie['title']} ({movie['year']})
        Genre: {movie['genre']}
        Rating: {movie['rating']}/10 ({movie['votes']} votes)
        Language: {movie['language']}
        Overview: {movie['overview']}
        """
        docs.append(Document(
            page_content=full_text,
            metadata={
                "title": movie["title"],
                "year": movie["year"],
                "genre": movie["genre"],
                "rating": movie["rating"],
                "votes": movie["votes"],
                "poster": movie["poster"] or "",
                "overview": movie["overview"]
            }
        ))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./moviematch_db",
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Load existing ChromaDB vector store from disk
    Use this if you've already built the store

    Returns:
        Existing ChromaDB vector store
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return Chroma(
        persist_directory="./moviematch_db",
        embedding_function=embeddings
    )


def semantic_search(query: str, vectorstore: Chroma, top_k: int = 5) -> tuple:
    """
    Perform semantic similarity search

    Args:
        query: User's search query
        vectorstore: ChromaDB vector store
        top_k: Number of results to return

    Returns:
        Tuple of (matches list, response_time_ms)
    """
    start = time.time()
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    elapsed = (time.time() - start) * 1000

    matches = []
    for doc, score in results:
        relevance = max(0, 1 - score) * 100
        matches.append({
            "title": doc.metadata["title"],
            "year": doc.metadata["year"],
            "genre": doc.metadata["genre"],
            "rating": doc.metadata["rating"],
            "votes": doc.metadata["votes"],
            "poster": doc.metadata["poster"],
            "overview": doc.metadata["overview"],
            "match_score": round(relevance, 2)
        })

    return matches, elapsed


# ============================================
# LLM Functions
# ============================================

def ask_gemini(question: str, context: str, api_key: str) -> str:
    """Use Google Gemini AI for movie Q&A"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""You are a movie expert. Answer based on this context.
        Context: {context}
        Question: {question}
        Answer in 2-3 sentences."""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


def ask_openai(question: str, context: str, api_key: str) -> str:
    """OpenAI - disabled for cost reasons"""
    return ("OpenAI integration disabled for cost reasons. "
            "Please use Gemini (free tier: 1500 req/day). "
            "Get key: https://aistudio.google.com/apikey")


def ask_claude(question: str, context: str, api_key: str) -> str:
    """Claude - disabled for cost reasons"""
    return ("Claude integration disabled for cost reasons. "
            "Please use Gemini (free tier: 1500 req/day). "
            "Get key: https://aistudio.google.com/apikey")


def ask_llm(question: str, context: str, api_key: str, provider: str = "gemini") -> str:
    """
    Provider-agnostic LLM router
    Swap providers with one parameter change

    Args:
        question: User question
        context: Movie context for grounding
        api_key: Provider API key
        provider: 'gemini' | 'openai' | 'claude'

    Returns:
        AI-generated answer string
    """
    if provider == "gemini":
        return ask_gemini(question, context, api_key)
    elif provider == "openai":
        return ask_openai(question, context, api_key)
    elif provider == "claude":
        return ask_claude(question, context, api_key)
    else:
        return f"Unknown provider: {provider}. Choose: gemini, openai, claude"
