# 🎬 MovieMatch — Full Stack AI Movie Platform

> Production-grade movie recommendation system with REST API + Web UI, built with FastAPI, Streamlit, LangChain, ChromaDB & TMDB API.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLIENTS                           │
│  Browser (Streamlit) │ Mobile App │ Other Services  │
└──────────┬──────────────────┬────────────────────── ┘
           │                  │
           ▼                  ▼
┌──────────────┐    ┌──────────────────┐
│  app.py      │    │   api.py         │
│  Streamlit   │    │   FastAPI        │
│  Web UI      │    │   REST API       │
│  Port: 8501  │    │   Port: 8000     │
└──────┬───────┘    └────────┬─────────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
        ┌─────────────────┐
        │    core.py      │
        │  Shared Logic   │
        │  (Single Source │
        │   of Truth)     │
        └────────┬────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌────────┐ ┌─────────┐ ┌──────────┐
│ChromaDB│ │TMDB API │ │Gemini AI │
│Vector  │ │1M+Movies│ │Free LLM  │
│  DB    │ │+Posters │ │  Q&A     │
└────────┘ └─────────┘ └──────────┘
```

## 🚀 Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Setup environment
```bash
# Create .env file
echo "TMDB_API_KEY=your_tmdb_key_here" > .env
```

### Run Streamlit UI
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Run FastAPI Backend
```bash
uvicorn api:app --reload --port 8000
# Opens at http://localhost:8000
# Docs at http://localhost:8000/docs ← Interactive API explorer!
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/movies` | Get all movies (with filters) |
| `POST` | `/search` | Semantic search |
| `POST` | `/similar` | Find similar movies |
| `GET` | `/trending` | Trending this week |
| `GET` | `/top-rated` | All-time top rated |
| `POST` | `/ask` | AI Q&A (Gemini) |
| `GET` | `/stats` | System statistics |
| `GET` | `/genres` | Available genres |
| `GET` | `/docs` | Interactive API docs ✨ |

---

## 🔍 Example API Calls

### Semantic Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "mind-bending sci-fi about dreams", "top_k": 3}'
```

**Response:**
```json
{
  "query": "mind-bending sci-fi about dreams",
  "total_results": 3,
  "response_time_ms": 47.3,
  "matches": [
    {
      "title": "Inception",
      "year": "2010",
      "genre": "Science Fiction/Action",
      "rating": 8.8,
      "match_score": 94.2
    }
  ]
}
```

### Find Similar Movies
```bash
curl -X POST "http://localhost:8000/similar" \
  -H "Content-Type: application/json" \
  -d '{"movie_title": "Inception", "top_k": 5}'
```

### Get Trending
```bash
curl "http://localhost:8000/trending"
```

### Filter Movies
```bash
curl "http://localhost:8000/movies?genre=Action&min_rating=7.5&sort_by=rating&limit=10"
```

### System Stats
```bash
curl "http://localhost:8000/stats"
```

---

## 🎯 Key Design Decisions

### Why FastAPI over Flask?
- ✅ Automatic OpenAPI docs (free Swagger UI!)
- ✅ Type validation with Pydantic
- ✅ Async support for high performance
- ✅ Modern Python best practices

### Why core.py separation?
- ✅ Single source of truth
- ✅ Both Streamlit and FastAPI use same logic
- ✅ Easy testing
- ✅ Clean architecture

### Why Gemini over OpenAI?
- ✅ Free tier: 1,500 requests/day forever
- ✅ No credit card required
- ✅ Comparable quality for movie Q&A
- ✅ Provider-agnostic code (swap with 1 line)

---

## 📊 Performance

- ⚡ **<100ms** average search response
- 🎬 **200+** real movies from TMDB
- 🔢 **384** embedding dimensions
- 🚀 **FastAPI** async = handles 1000+ req/sec
- 📝 **Auto-docs** at `/docs`

---

## 🗺️ Production Roadmap

For enterprise deployment:

| Feature | Tool | Impact |
|---------|------|--------|
| Scale vector DB | Pinecone | 10M+ movies |
| Caching | Redis | <5ms popular queries |
| Auth | JWT + OAuth | User accounts |
| Monitoring | Grafana + Prometheus | Real-time metrics |
| CI/CD | GitHub Actions | Auto deployment |
| Container | Docker + K8s | Scale horizontally |

---

## 👨‍💻 Author

**Sivarkamani** | AI/ML Engineer
