"""
MovieMatch: AI-Powered Movie Recommendation System
Full TMDB Integration with 1000+ Real Movies

Author: Sivarkamani
Tech Stack: Streamlit, LangChain, ChromaDB, Sentence Transformers, TMDB API, Gemini AI
"""

import streamlit as st
import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Optional: Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ============================================
# Constants
# ============================================
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"
TOTAL_PAGES = 10  # 10 pages x 20 movies = 200 movies

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="MovieMatch - AI Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #E50914 0%, #831010 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .match-score {
        background: #FFD700;
        color: #000;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .stats-box {
        background: linear-gradient(135deg, #E50914 0%, #831010 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .movie-poster {
        border-radius: 8px;
        width: 100%;
    }
    .trending-badge {
        background: #E50914;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .genre-badge {
        background: #333;
        color: #FFD700;
        padding: 0.2rem 0.6rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# TMDB API Functions
# ============================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_tmdb_movies(total_pages=TOTAL_PAGES):
    """
    Fetch movies from TMDB API
    - Popular movies across multiple pages
    - Returns rich movie data with posters, cast, genres
    """
    if not TMDB_API_KEY:
        st.error("❌ TMDB API key not found! Check your .env file.")
        return []

    movies = []
    genre_map = fetch_genres()

    progress = st.progress(0, text="🎬 Fetching movies from TMDB...")

    seen_ids = set()  # Track seen movies to avoid duplicates

    for page in range(1, total_pages + 1):
        try:
            # Fetch popular movies
            url = f"{TMDB_BASE_URL}/movie/popular"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "page": page
            }
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                for movie in data.get("results", []):
                    # Skip duplicates
                    movie_id = movie.get("id")
                    if movie_id in seen_ids:
                        continue
                    seen_ids.add(movie_id)

                    # Get genre names
                    genre_names = [
                        genre_map.get(gid, "Unknown")
                        for gid in movie.get("genre_ids", [])[:3]
                    ]

                    movies.append({
                        "id": movie_id,
                        "title": movie.get("title", "Unknown"),
                        "year": movie.get("release_date", "")[:4] if movie.get("release_date") else "N/A",
                        "rating": round(movie.get("vote_average", 0), 1),
                        "votes": movie.get("vote_count", 0),
                        "overview": movie.get("overview", "No description available"),
                        "genre": "/".join(genre_names) if genre_names else "Unknown",
                        "poster": f"{TMDB_IMAGE_BASE}{movie.get('poster_path')}" if movie.get("poster_path") else None,
                        "popularity": movie.get("popularity", 0),
                        "language": movie.get("original_language", "en")
                    })

            # Update progress
            progress.progress(page / total_pages, text=f"🎬 Loading page {page}/{total_pages}...")
            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            st.warning(f"⚠️ Error fetching page {page}: {str(e)}")
            continue

    progress.empty()
    return movies


@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_genres():
    """Fetch genre mapping from TMDB"""
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


@st.cache_data(ttl=3600)
def fetch_trending():
    """Fetch trending movies this week"""
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


@st.cache_data(ttl=3600)
def fetch_top_rated():
    """Fetch top rated movies"""
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
# Vector Store Setup (Cached)
# ============================================
@st.cache_resource
def setup_vectorstore(movies_json):
    """
    Build ChromaDB vector store from TMDB movies
    Cached so it only builds once per session
    """
    import json
    _movies = json.loads(movies_json)
    
    if not _movies:
        return None, 0

    docs = []
    for movie in _movies:
        # Rich document combining all movie metadata
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

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Build ChromaDB
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./moviematch_tmdb_db",
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vectorstore, len(docs)


# ============================================
# Search Function
# ============================================
def search_movies(query, vectorstore, top_k=5):
    """Semantic search with deduplication"""
    start = time.time()
    # Fetch more to account for duplicates
    results = vectorstore.similarity_search_with_score(query, k=top_k * 3)
    elapsed = (time.time() - start) * 1000

    matches = []
    seen_titles = set()  # Deduplicate results

    for doc, score in results:
        title = doc.metadata["title"]

        # Skip duplicate titles
        if title in seen_titles:
            continue
        seen_titles.add(title)

        relevance = max(0, 1 - score) * 100
        matches.append({
            "title": title,
            "year": doc.metadata["year"],
            "genre": doc.metadata["genre"],
            "rating": doc.metadata["rating"],
            "votes": doc.metadata["votes"],
            "poster": doc.metadata["poster"],
            "overview": doc.metadata["overview"],
            "match_score": round(relevance, 2)
        })

        # Stop when we have enough unique results
        if len(matches) >= top_k:
            break

    return matches, elapsed


# ============================================
# Multi-Provider LLM Integration
# ============================================
def ask_gemini(question, context, api_key):
    """Use Google Gemini AI"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""You are a movie expert. Answer the user's question based on this movie information.
        
        Movie Context: {context}
        Question: {question}
        
        Answer in 2-3 sentences."""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}. Please check your Gemini API key."


def ask_openai(question, context, api_key):
    """OpenAI - disabled for cost reasons"""
    return ("⚠️ **OpenAI is disabled for cost reasons.**\n\n"
            "💡 Please choose **Gemini** instead:\n"
            "- ✅ 1,500 FREE requests/day forever\n"
            "- ✅ No credit card required\n\n"
            "🔗 Get free key: https://aistudio.google.com/apikey")


def ask_claude(question, context, api_key):
    """Claude - disabled for cost reasons"""
    return ("⚠️ **Claude is disabled for cost reasons.**\n\n"
            "💡 Please choose **Gemini** instead:\n"
            "- ✅ 1,500 FREE requests/day forever\n"
            "- ✅ No credit card required\n\n"
            "🔗 Get free key: https://aistudio.google.com/apikey")


def ask_llm(question, context, api_key, provider="gemini"):
    """
    Multi-provider LLM router
    Provider-agnostic architecture — swap with one line change
    """
    if provider == "gemini":
        return ask_gemini(question, context, api_key)
    elif provider == "openai":
        return ask_openai(question, context, api_key)
    elif provider == "claude":
        return ask_claude(question, context, api_key)
    else:
        return "❌ Unknown provider. Please choose: gemini, openai, or claude."


# ============================================
# Display Movie Card with Poster
# ============================================
def display_movie_card(match, rank, show_ai=False, api_key=None, provider="gemini"):
    """Display a beautiful movie card with poster"""
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        st.markdown(f"### #{rank}")
        st.markdown(
            f'<span class="match-score">{match["match_score"]:.1f}%</span>',
            unsafe_allow_html=True
        )

    with col2:
        if match.get("poster"):
            st.image(match["poster"], width=100)
        else:
            st.markdown("🎬")

    with col3:
        st.markdown(f"**🎬 {match['title']} ({match['year']})**")
        st.markdown(f"⭐ {match['rating']}/10 | 🗳️ {match['votes']:,} votes | 🎭 {match['genre']}")

        with st.expander("📖 Overview + AI Q&A"):
            st.markdown(match["overview"])
            st.markdown("---")

            # Unique key for this movie
            a_key = f"ans_{rank}_{hash(match['title'])}"

            if not api_key:
                st.warning("🔑 Enter Gemini API key in sidebar!")
            elif not show_ai:
                st.info("💡 Check Enable AI Q&A in sidebar!")
            else:
                st.success("🤖 AI Ready!")

                # Use st.form to prevent page jumping
                with st.form(key=f"form_{rank}_{hash(match['title'])}"):
                    user_q = st.text_input(
                        f"💬 Ask about {match['title']}:",
                        placeholder="What themes does this movie explore?"
                    )
                    submitted = st.form_submit_button(
                        "🤖 Ask Gemini",
                        type="primary"
                    )

                # Handle submission OUTSIDE form
                if submitted and user_q:
                    with st.spinner("🤖 Gemini is thinking..."):
                        answer = ask_llm(
                            user_q,
                            match["overview"],
                            api_key,
                            provider
                        )
                    st.session_state[a_key] = answer

                elif submitted and not user_q:
                    st.warning("Please type a question!")

                # Show answer if exists
                if a_key in st.session_state and st.session_state[a_key]:
                    st.markdown("**🤖 Gemini says:**")
                    st.info(st.session_state[a_key])

    st.markdown("---")


# ============================================
# Main App
# ============================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 MovieMatch</h1>
        <h3>AI-Powered Movie Discovery using RAG + TMDB</h3>
        <p>Search across 200+ real movies by vibe, mood, or theme</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Movie fetch settings
        st.subheader("🎬 Database Settings")
        pages = st.slider(
            "Movies to load (20 per page):",
            min_value=1,
            max_value=20,
            value=10,
            help="More pages = more movies but slower loading"
        )
        total_movies_estimate = pages * 20
        st.info(f"📊 Will load ~{total_movies_estimate} movies")

        top_k = st.slider("Search results to show:", 1, 10, 5)

        st.markdown("---")

        # Genre filter
        st.subheader("🎭 Filter by Genre")
        genres = ["All", "Action", "Comedy", "Drama", "Horror",
                  "Science Fiction", "Romance", "Thriller", "Animation",
                  "Adventure", "Crime", "Fantasy"]
        selected_genre = st.selectbox("Genre:", genres)

        st.markdown("---")

        # Multi-Provider AI Settings
        st.subheader("🤖 AI Chat (Optional)")

        if GEMINI_AVAILABLE:
            provider = st.selectbox(
                "Choose AI Provider:",
                ["gemini", "openai", "claude"],
                format_func=lambda x: {
                    "gemini": "✅ Gemini (Free)",
                    "openai": "💰 OpenAI (Paid)",
                    "claude": "💰 Claude (Paid)"
                }[x]
            )

            if provider != "gemini":
                st.warning(
                    f"⚠️ **Cost Alert!**\n\n"
                    f"Please choose **Gemini** instead.\n"
                    f"{provider.capitalize()} requires paid access.\n"
                    f"Gemini = 1,500 FREE requests/day!"
                )
            else:
                st.success("✅ Great choice! Gemini is free.")

            # Session state keeps key during session
            # Clears when browser closes (security!)
            if "api_key" not in st.session_state:
                st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")

            api_key = st.text_input(
                f"{provider.capitalize()} API Key:",
                type="password",
                value=st.session_state.api_key,
                help="Get free Gemini key: https://aistudio.google.com/apikey"
            )

            # Save to session state
            if api_key:
                st.session_state.api_key = api_key
                st.success("✅ Key saved for this session!")
            else:
                st.info("💡 Enter key once per session - survives refresh!")

            use_ai = st.checkbox("Enable AI Q&A", value=False)
        else:
            st.warning("Install google-generativeai for AI features")
            api_key = None
            use_ai = False
            provider = "gemini"

        st.markdown("---")
        st.subheader("📊 About")
        st.markdown("""
        **Tech Stack:**
        - 🎬 TMDB API (Real movies)
        - 🦜 LangChain (RAG Pipeline)
        - 🔮 ChromaDB (Vector DB)
        - 🤗 Sentence Transformers
        - ⚡ Streamlit (UI)
        - 🤖 Gemini AI (Optional)
        """)
        st.markdown("**Built by:** Sivarkamani")

    # ============================================
    # Load Movies from TMDB
    # ============================================
    with st.spinner("🎬 Connecting to TMDB and loading movies..."):
        movies = fetch_tmdb_movies(total_pages=pages)

    if not movies:
        st.error("❌ Failed to load movies. Check your TMDB API key in .env file!")
        return

    # Apply genre filter
    if selected_genre != "All":
        movies = [m for m in movies if selected_genre.lower() in m["genre"].lower()]

    # Build vector store
    with st.spinner("🧠 Building AI search engine..."):
        import json
        movies_json = json.dumps(movies)
        vectorstore, total_indexed = setup_vectorstore(movies_json)

    if not vectorstore:
        st.error("❌ Failed to build search engine!")
        return

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stats-box"><h3>{len(movies)}</h3><p>Real Movies</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stats-box"><h3>TMDB</h3><p>Live Database</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stats-box"><h3><100ms</h3><p>Search Speed</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stats-box"><h3>🎬</h3><p>With Posters!</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ============================================
    # Tabs
    # ============================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Search",
        "🔥 Trending",
        "⭐ Top Rated",
        "📊 Analytics"
    ])

    # ============================================
    # Tab 1: Semantic Search
    # ============================================
    with tab1:
        st.header("🔍 Search Movies by Vibe")
        st.markdown("*Describe what you feel like watching — not the title!*")

        # Example queries
        st.markdown("**💡 Try these:**")
        col1, col2, col3, col4 = st.columns(4)
        examples = [
            "mind-bending sci-fi",
            "feel-good comedy",
            "intense psychological thriller",
            "epic adventure"
        ]
        if "query" not in st.session_state:
            st.session_state.query = ""

        for i, example in enumerate(examples):
            col = [col1, col2, col3, col4][i]  # Keep 0-based for list index
            if col.button(f"💡 {example}", key=f"ex_{i}"):
                st.session_state.query = example

        query = st.text_input(
            "Or describe your perfect movie:",
            value=st.session_state.query,
            placeholder="e.g., heartwarming story about family and sacrifice..."
        )

        if st.button("🎬 Find My Movie!", type="primary") and query:
            with st.spinner("🔮 Searching with AI..."):
                matches, response_time = search_movies(query, vectorstore, top_k)

            st.success(f"⚡ Found {len(matches)} matches in {response_time:.1f}ms")
            st.markdown("### 🎯 Best Matches:")

            for i, match in enumerate(matches, 1):
                display_movie_card(
                    match, i,
                    show_ai=use_ai,
                    api_key=api_key,
                    provider=provider
                )

    # ============================================
    # Tab 2: Trending
    # ============================================
    with tab2:
        st.header("🔥 Trending This Week")
        st.markdown("*Real-time trending data from TMDB*")

        trending = fetch_trending()

        if trending:
            cols = st.columns(5)
            for i, movie in enumerate(trending[:5], 1):
                with cols[i-1]:
                    if movie.get("poster"):
                        st.image(movie["poster"], use_container_width=True)
                    st.markdown(f"**{movie['title']}**")
                    st.markdown(f"⭐ {movie['rating']} | {movie['year']}")
                    st.markdown(f'<span class="trending-badge">🔥 Trending</span>', unsafe_allow_html=True)

            st.markdown("---")
            cols2 = st.columns(5)
            for i, movie in enumerate(trending[5:10], 1):
                with cols2[i-1]:
                    if movie.get("poster"):
                        st.image(movie["poster"], use_container_width=True)
                    st.markdown(f"**{movie['title']}**")
                    st.markdown(f"⭐ {movie['rating']} | {movie['year']}")
        else:
            st.warning("Could not load trending movies. Check API key.")

    # ============================================
    # Tab 3: Top Rated
    # ============================================
    with tab3:
        st.header("⭐ Top Rated Movies")
        st.markdown("*Highest rated movies of all time from TMDB*")

        top_rated = fetch_top_rated()

        if top_rated:
            cols = st.columns(5)
            for i, movie in enumerate(top_rated[:5], 1):
                with cols[i-1]:
                    if movie.get("poster"):
                        st.image(movie["poster"], use_container_width=True)
                    st.markdown(f"**{movie['title']}**")
                    st.markdown(f"⭐ {movie['rating']} | {movie['year']}")

            st.markdown("---")
            cols2 = st.columns(5)
            for i, movie in enumerate(top_rated[5:10], 1):
                with cols2[i-1]:
                    if movie.get("poster"):
                        st.image(movie["poster"], use_container_width=True)
                    st.markdown(f"**{movie['title']}**")
                    st.markdown(f"⭐ {movie['rating']} | {movie['year']}")
        else:
            st.warning("Could not load top rated movies. Check API key.")

    # ============================================
    # Tab 4: Analytics
    # ============================================
    with tab4:
        st.header("📊 Database Analytics")

        df = pd.DataFrame(movies)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⭐ Rating Distribution")
            rating_bins = pd.cut(df["rating"], bins=[0, 4, 6, 7, 8, 10],
                                  labels=["<4", "4-6", "6-7", "7-8", "8+"])
            st.bar_chart(rating_bins.value_counts().sort_index())

        with col2:
            st.subheader("🎭 Top Genres")
            df["main_genre"] = df["genre"].str.split("/").str[0]
            genre_counts = df["main_genre"].value_counts().head(10)
            st.bar_chart(genre_counts)
            
            # Genre table with proper numbering
            genre_df = genre_counts.reset_index()
            genre_df.columns = ["Genre", "Count"]
            genre_df.index = range(1, len(genre_df) + 1)
            genre_df.index.name = "No."
            st.dataframe(genre_df, use_container_width=True)

        st.subheader("📅 Movies by Year")
        year_counts = df[df["year"] != "N/A"]["year"].value_counts().sort_index()
        st.bar_chart(year_counts)
        
        st.subheader("🏆 Top 10 Movies by Rating")
        top10 = df.nlargest(10, "rating")[["title", "rating", "year", "genre"]].reset_index(drop=True)
        top10.index = range(1, len(top10) + 1)
        top10.index.name = "Rank"
        st.dataframe(top10, use_container_width=True)

        st.subheader("🎬 Full Movie Database")
        df_display = df[["title", "year", "genre", "rating", "votes"]].sort_values("rating", ascending=False).reset_index(drop=True)
        df_display.index = range(1, len(df_display) + 1)
        df_display.index.name = "No."
        st.dataframe(df_display, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        🎬 MovieMatch | Powered by TMDB API + LangChain + ChromaDB + Streamlit
        <br>Real movies. Real AI. Real recommendations.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
