"""
Book Recommender – Flask REST API
==================================
Endpoints:
  GET  /api/health                          – liveness probe
  GET  /api/books/search?q=<query>         – title search
  GET  /api/books/<isbn>                    – book detail
  GET  /api/genres                          – list genres
  GET  /api/stats                           – dataset statistics
  POST /api/recommend/title                 – recs by title
  POST /api/recommend/isbn                  – recs by ISBN
  POST /api/recommend/description           – recs by free text
  GET  /                                    – serve frontend
"""

import logging
import os
from functools import lru_cache

from flask import Flask, jsonify, render_template_string, request, send_from_directory
try:
    from flask_cors import CORS
    _CORS_AVAILABLE = True
except ImportError:
    _CORS_AVAILABLE = False

from recommender import BookRecommender, DataLoader
from recommender.model import ModelConfig

# ──────────────────────────────────────────────────────────────────────── #
#  Logging                                                                  #
# ──────────────────────────────────────────────────────────────────────── #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────── #
#  App factory                                                              #
# ──────────────────────────────────────────────────────────────────────── #


def create_app(data_path: str | None = None, config: ModelConfig | None = None) -> Flask:
    app = Flask(__name__, static_folder="frontend", static_url_path="")
    if _CORS_AVAILABLE:
        CORS(app)

    # ── Initialise model ────────────────────────────────────────────────
    loader = DataLoader(data_path)
    df = loader.load()
    recommender = BookRecommender(config)
    recommender.fit(df)
    stats_cache = loader.stats()
    logger.info("Model ready. %d books loaded.", len(df))

    # ── Helpers ─────────────────────────────────────────────────────────

    def ok(payload, status: int = 200):
        return jsonify({"status": "ok", "data": payload}), status

    def err(message: str, status: int = 400):
        return jsonify({"status": "error", "message": message}), status

    # ── Routes ──────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return send_from_directory("frontend", "index.html")

    @app.route("/api/health")
    def health():
        return ok({"healthy": True, "books_loaded": len(df)})

    @app.route("/api/stats")
    def stats():
        return ok(stats_cache)

    @app.route("/api/genres")
    def genres():
        return ok(recommender.list_genres())

    @app.route("/api/books/search")
    def search_books():
        query = request.args.get("q", "").strip()
        if not query:
            return err("Query parameter 'q' is required.")
        limit = min(int(request.args.get("limit", 10)), 50)
        titles = recommender.search_titles(query, limit=limit)
        return ok(titles)

    @app.route("/api/books/<path:isbn>")
    def book_detail(isbn: str):
        book = None
        # Try ISBN lookup via search_titles fallback
        row_matches = df[df["isbn"] == isbn]
        if not row_matches.empty:
            row = row_matches.iloc[0]
            book = {
                "isbn": str(row.get("isbn", "")),
                "title": str(row.get("title", "")),
                "author": str(row.get("author", "")),
                "genre": str(row.get("genre", "")),
                "subgenre": str(row.get("subgenre", "")),
                "description": str(row.get("description", "")),
                "avg_rating": float(row.get("avg_rating") or 0),
                "num_ratings": int(row.get("num_ratings") or 0),
                "year": int(row["year"]) if str(row.get("year", "")) not in ("", "nan") else None,
                "pages": int(row["pages"]) if str(row.get("pages", "")) not in ("", "nan") else None,
                "language": str(row.get("language", "English")),
            }
        if book is None:
            return err(f"Book with ISBN '{isbn}' not found.", 404)
        return ok(book)

    @app.route("/api/recommend/title", methods=["POST"])
    def recommend_by_title():
        body = request.get_json(silent=True) or {}
        title = (body.get("title") or "").strip()
        if not title:
            return err("'title' field is required.")

        top_n = min(int(body.get("top_n", 10)), 30)
        filter_genre = body.get("filter_genre") or None
        filter_author = body.get("filter_author") or None

        try:
            results = recommender.recommend_by_title(
                title,
                top_n=top_n,
                filter_genre=filter_genre,
                filter_author=filter_author,
            )
        except ValueError as exc:
            return err(str(exc), 404)

        query_book = recommender.get_book_by_title(title)
        return ok({
            "query_book": query_book,
            "recommendations": [r.to_dict() for r in results],
            "count": len(results),
        })

    @app.route("/api/recommend/isbn", methods=["POST"])
    def recommend_by_isbn():
        body = request.get_json(silent=True) or {}
        isbn = (body.get("isbn") or "").strip()
        if not isbn:
            return err("'isbn' field is required.")

        top_n = min(int(body.get("top_n", 10)), 30)
        filter_genre = body.get("filter_genre") or None

        try:
            results = recommender.recommend_by_isbn(isbn, top_n=top_n, filter_genre=filter_genre)
        except ValueError as exc:
            return err(str(exc), 404)

        return ok({
            "recommendations": [r.to_dict() for r in results],
            "count": len(results),
        })

    @app.route("/api/recommend/description", methods=["POST"])
    def recommend_by_description():
        body = request.get_json(silent=True) or {}
        description = (body.get("description") or "").strip()
        if not description:
            return err("'description' field is required.")
        if len(description) < 10:
            return err("Description too short – please provide at least 10 characters.")

        top_n = min(int(body.get("top_n", 10)), 30)

        results = recommender.recommend_by_description(description, top_n=top_n)
        return ok({
            "recommendations": [r.to_dict() for r in results],
            "count": len(results),
        })

    return app


# ──────────────────────────────────────────────────────────────────────── #
#  Entrypoint                                                               #
# ──────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=debug)
