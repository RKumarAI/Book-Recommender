#  Bibliograph — Book Recommender

A **content-based filtering** system that recommends books based on descriptions, genres, subgenres, and author style. No ratings history or user accounts required — just tell it what you like and it finds similar books.
---

##  Features

| Feature | Description |
|---|---|
| **TF-IDF Text Matching** | Analyses book descriptions, genres, and author names with n-gram support |
| **Genre Encoding** | One-hot encoded genre features with configurable weighting |
| **Rating Boost** | Optional normalised rating feature to surface highly-rated similar books |
| **Three Recommendation Modes** | By title · By ISBN · By free-text description |
| **REST API** | Flask backend with CORS, input validation, and JSON responses |
| **Web UI** | Editorial-style dark frontend with autocomplete and genre filtering |
| **CLI** | Coloured terminal interface for quick exploration |
| **Configurable** | Swap datasets, tune hyperparameters, persist models to disk |

---

## Project Structure

```
book-recommender/
├── app.py                   # Flask REST API server
├── cli.py                   # Command-line interface
├── requirements.txt
│
├── recommender/
│   ├── __init__.py
│   ├── model.py             # BookRecommender (TF-IDF + cosine similarity)
│   └── data_loader.py       # CSV loading, validation & preprocessing
│
├── data/
│   └── books.csv            # 100+ book dataset (ready to extend)
│
├── frontend/
│   └── index.html           # Single-file web UI
│
└── tests/
    └── test_recommender.py  # 35+ unit & integration tests
```

---

##  Quick Start

### 1 — Install dependencies

```bash
git clone https://github.com/yourname/book-recommender.git
cd book-recommender
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Start the web server

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### 3 — Use the CLI

```bash
# Recommend books similar to a title
python cli.py recommend --title "The Hunger Games" --top-n 6

# Recommend by description
python cli.py recommend --description "A dystopian society where children fight to the death"

# Filter by genre
python cli.py recommend --title "Dune" --genre "Science Fiction"

# Search for titles
python cli.py search "tolkien"

# Dataset info
python cli.py stats
python cli.py genres
```

---

##  REST API Reference

All endpoints return `{ "status": "ok", "data": ... }` on success or `{ "status": "error", "message": "..." }` on failure.

### `GET /api/health`
Liveness probe.

### `GET /api/stats`
Dataset statistics: total books, genre count, author count, average rating.

### `GET /api/genres`
Sorted list of all genres.

### `GET /api/books/search?q=<query>&limit=10`
Title autocomplete search.

### `GET /api/books/<isbn>`
Full book detail by ISBN.

### `POST /api/recommend/title`
```json
{
  "title": "The Hunger Games",
  "top_n": 8,
  "filter_genre": "Fiction",   // optional
  "filter_author": "Collins"   // optional – exclude this author
}
```

### `POST /api/recommend/isbn`
```json
{
  "isbn": "9780061965579",
  "top_n": 8,
  "filter_genre": "Fiction"    // optional
}
```

### `POST /api/recommend/description`
```json
{
  "description": "A girl must survive a government-controlled arena while sparking a revolution",
  "top_n": 8
}
```

---

##  How It Works

### Feature Pipeline

```
Book metadata
     │
     ▼
┌─────────────────────────────┐
│  1. Text features           │  TF-IDF (bigrams) on:
│     description             │  description + genre + subgenre + author
│     genre × 3 repetitions   │  (genre/subgenre upweighted via repetition)
│     subgenre × 2            │
│     author                  │
└────────────────┬────────────┘
                 │
┌────────────────▼────────────┐
│  2. Genre one-hot           │  OneHotEncoder on top-50 genres
│     × genre_weight (2.0)    │  Weighted to strengthen genre similarity
└────────────────┬────────────┘
                 │
┌────────────────▼────────────┐
│  3. Rating feature          │  MinMax-normalised avg_rating
│     × rating_weight (0.5)   │  Optional, configurable
└────────────────┬────────────┘
                 │
                 ▼
        Sparse feature matrix (N × F)
                 │
                 ▼
        Cosine similarity → ranked recommendations
```

### Similarity Score

Cosine similarity measures the angle between two book vectors:

```
similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

Score ranges from 0 (no similarity) to 1 (identical). Displayed as a percentage in the UI.

---

##  Configuration

Customise via `ModelConfig`:

```python
from recommender.model import ModelConfig, BookRecommender

config = ModelConfig(
    tfidf_max_features=20_000,
    tfidf_ngram_range=(1, 3),     # up to trigrams
    genre_weight=3.0,              # stronger genre matching
    use_rating_feature=True,
    rating_weight=1.0,
    default_top_n=10,
    min_similarity=0.05,
)

rec = BookRecommender(config=config)
```

---

##  Using Your Own Dataset

The system accepts any CSV with these columns:

| Column | Required | Description |
|---|---|---|
| `isbn` | ✅ | Unique book identifier |
| `title` | ✅ | Book title |
| `author` | ✅ | Author name(s) |
| `genre` | ✅ | Primary genre |
| `description` | ✅ | Book synopsis (longer = better recommendations) |
| `subgenre` | ☑️ | More specific genre tag |
| `year` | ☑️ | Publication year |
| `avg_rating` | ☑️ | Average rating (e.g. 1–5 scale) |
| `num_ratings` | ☑️ | Number of ratings |
| `pages` | ☑️ | Page count |
| `language` | ☑️ | Language |

Pass your CSV path to `DataLoader` and `create_app`:

```python
# Python
from recommender import DataLoader, BookRecommender
loader = DataLoader("path/to/my_books.csv")
df = loader.load()
rec = BookRecommender()
rec.fit(df)

# Flask server
from app import create_app
app = create_app(data_path="path/to/my_books.csv")
```

Or set via environment variable:

```bash
DATA_PATH=data/my_books.csv python app.py
```

---

##  Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=recommender --cov-report=term-missing

# Specific test class
pytest tests/ -k "TestFlaskAPI" -v
```

---

##  Development

```bash
# Enable debug mode (auto-reload)
FLASK_DEBUG=true python app.py

# Save a trained model to disk
python -c "
from recommender import DataLoader, BookRecommender
rec = BookRecommender()
rec.fit(DataLoader().load())
rec.save('models/recommender.pkl')
print('Saved!')
"

# Load and use saved model
python -c "
from recommender.model import BookRecommender
rec = BookRecommender.load('models/recommender.pkl')
for r in rec.recommend_by_title('Dune', top_n=3):
    print(r.title, round(r.similarity_score, 3))
"
```



