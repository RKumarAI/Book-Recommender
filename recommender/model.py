"""
Content-Based Filtering Model for Book Recommendations.

Feature pipeline
----------------
1.  Text features  – TF-IDF on a concatenation of description, genre,
                     subgenre, and author name.
2.  Genre features – one-hot encoding of the top-N genres.
3.  Rating feature – normalised average rating (optional, configurable).

Similarity metric: cosine similarity (dense cosine on the full matrix;
optionally falls back to sparse top-k for large datasets).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────── #
#  Data structures                                                          #
# ──────────────────────────────────────────────────────────────────────── #


@dataclass
class RecommendedBook:
    """A single book recommendation with its similarity score."""

    isbn: str
    title: str
    author: str
    genre: str
    subgenre: str
    description: str
    avg_rating: float
    num_ratings: int
    year: Optional[int]
    pages: Optional[int]
    language: str
    similarity_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "isbn": self.isbn,
            "title": self.title,
            "author": self.author,
            "genre": self.genre,
            "subgenre": self.subgenre,
            "description": self.description,
            "avg_rating": self.avg_rating,
            "num_ratings": self.num_ratings,
            "year": self.year,
            "pages": self.pages,
            "language": self.language,
            "similarity_score": round(self.similarity_score, 4),
        }


@dataclass
class ModelConfig:
    """Hyperparameters for the recommender."""

    # TF-IDF
    tfidf_max_features: int = 15_000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_min_df: int = 1
    tfidf_sublinear_tf: bool = True

    # Genre one-hot
    max_genre_categories: int = 50
    genre_weight: float = 2.0          # multiplier on one-hot genre matrix

    # Rating boost
    use_rating_feature: bool = True
    rating_weight: float = 0.5

    # Inference
    default_top_n: int = 10
    min_similarity: float = 0.01


# ──────────────────────────────────────────────────────────────────────── #
#  Feature construction helpers                                             #
# ──────────────────────────────────────────────────────────────────────── #


def _build_text_corpus(df: pd.DataFrame) -> pd.Series:
    """Combine relevant text columns into one document per book."""
    parts = [
        df.get("description", pd.Series("", index=df.index)).fillna(""),
        df.get("genre", pd.Series("", index=df.index)).fillna("").apply(
            lambda g: " ".join([g] * 3)  # upweight genre
        ),
        df.get("subgenre", pd.Series("", index=df.index)).fillna("").apply(
            lambda s: " ".join([s] * 2)
        ),
        df.get("author", pd.Series("", index=df.index)).fillna(""),
    ]
    return parts[0].str.cat(parts[1:], sep=" ").str.lower()


# ──────────────────────────────────────────────────────────────────────── #
#  Main recommender class                                                   #
# ──────────────────────────────────────────────────────────────────────── #


class BookRecommender:
    """
    Content-based book recommender using TF-IDF + genre features.

    Usage
    -----
    >>> from recommender import BookRecommender, DataLoader
    >>> loader = DataLoader()
    >>> df = loader.load()
    >>> rec = BookRecommender()
    >>> rec.fit(df)
    >>> results = rec.recommend_by_title("The Hunger Games", top_n=5)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self._df: Optional[pd.DataFrame] = None
        self._feature_matrix: Optional[Any] = None  # sparse or dense
        self._tfidf: Optional[TfidfVectorizer] = None
        self._genre_encoder: Optional[OneHotEncoder] = None
        self._scaler: Optional[MinMaxScaler] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    #  Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> "BookRecommender":
        """
        Build the feature matrix from a books DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned books data as returned by DataLoader.
        """
        logger.info("Fitting recommender on %d books …", len(df))
        self._df = df.reset_index(drop=True)
        cfg = self.config

        # 1. TF-IDF on text
        corpus = _build_text_corpus(self._df)
        self._tfidf = TfidfVectorizer(
            max_features=cfg.tfidf_max_features,
            ngram_range=cfg.tfidf_ngram_range,
            min_df=cfg.tfidf_min_df,
            sublinear_tf=cfg.tfidf_sublinear_tf,
        )
        tfidf_mat = self._tfidf.fit_transform(corpus)  # sparse (N, V)
        logger.debug("TF-IDF shape: %s", tfidf_mat.shape)

        # 2. One-hot genre
        genre_col = self._df["genre"].fillna("Unknown").values.reshape(-1, 1)
        self._genre_encoder = OneHotEncoder(
            max_categories=cfg.max_genre_categories,
            sparse_output=True,
            handle_unknown="ignore",
        )
        genre_mat = self._genre_encoder.fit_transform(genre_col)
        genre_mat = genre_mat * cfg.genre_weight
        logger.debug("Genre one-hot shape: %s", genre_mat.shape)

        # 3. Optional rating feature
        matrices = [tfidf_mat, genre_mat]
        if cfg.use_rating_feature and "rating_norm" in self._df.columns:
            ratings = self._df["rating_norm"].fillna(0.0).values.reshape(-1, 1)
            self._scaler = MinMaxScaler()
            rating_scaled = self._scaler.fit_transform(ratings) * cfg.rating_weight
            matrices.append(csr_matrix(rating_scaled))

        # 4. Combine
        self._feature_matrix = hstack(matrices, format="csr")
        logger.info(
            "Feature matrix built: %s (nnz=%d)",
            self._feature_matrix.shape,
            self._feature_matrix.nnz,
        )
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  Recommendation entry-points                                          #
    # ------------------------------------------------------------------ #

    def recommend_by_title(
        self,
        title: str,
        top_n: Optional[int] = None,
        filter_genre: Optional[str] = None,
        filter_author: Optional[str] = None,
    ) -> List[RecommendedBook]:
        """
        Return the top-N most similar books for a given title.

        Parameters
        ----------
        title       : Book title (case-insensitive, partial match supported).
        top_n       : Number of recommendations (default from config).
        filter_genre: If given, restrict results to this genre.
        filter_author: If given, exclude results by this author.
        """
        self._check_fitted()
        idx = self._resolve_title(title)
        return self._get_recommendations(
            idx,
            top_n=top_n or self.config.default_top_n,
            filter_genre=filter_genre,
            filter_author=filter_author,
        )

    def recommend_by_isbn(
        self,
        isbn: str,
        top_n: Optional[int] = None,
        filter_genre: Optional[str] = None,
    ) -> List[RecommendedBook]:
        """Return recommendations for a book identified by ISBN."""
        self._check_fitted()
        idx = self._resolve_isbn(isbn)
        return self._get_recommendations(
            idx, top_n=top_n or self.config.default_top_n, filter_genre=filter_genre
        )

    def recommend_by_description(
        self,
        description: str,
        top_n: Optional[int] = None,
    ) -> List[RecommendedBook]:
        """
        Recommend books based on an arbitrary free-text description.
        Useful for 'I want a book about …' queries.
        """
        self._check_fitted()
        cfg = self.config

        # Transform query through the same TF-IDF
        query_tfidf = self._tfidf.transform([description.lower()])

        # Pad with zeros for genre + rating columns
        n_extra = self._feature_matrix.shape[1] - query_tfidf.shape[1]
        padding = csr_matrix((1, n_extra))
        query_vec = hstack([query_tfidf, padding], format="csr")

        scores = cosine_similarity(query_vec, self._feature_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][: (top_n or cfg.default_top_n)]
        return [
            self._build_result(i, scores[i])
            for i in top_idx
            if scores[i] >= cfg.min_similarity
        ]

    def get_book_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Return book metadata for a given title."""
        self._check_fitted()
        try:
            idx = self._resolve_title(title)
            row = self._df.iloc[idx]
            return self._row_to_dict(row)
        except ValueError:
            return None

    def search_titles(self, query: str, limit: int = 10) -> List[str]:
        """Return titles that contain the query string (case-insensitive)."""
        self._check_fitted()
        mask = self._df["title"].str.contains(query, case=False, na=False)
        return self._df.loc[mask, "title"].head(limit).tolist()

    def list_genres(self) -> List[str]:
        """Return sorted list of unique genres in the dataset."""
        self._check_fitted()
        return sorted(self._df["genre"].dropna().unique().tolist())

    # ------------------------------------------------------------------ #
    #  Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Pickle the fitted recommender to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Recommender saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BookRecommender":
        """Load a previously saved recommender from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected BookRecommender, got {type(obj)}")
        logger.info("Recommender loaded from %s", path)
        return obj

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if not self._fitted or self._df is None:
            raise RuntimeError(
                "Model is not fitted. Call .fit(df) before requesting recommendations."
            )

    def _resolve_title(self, title: str) -> int:
        """Return row index of the best-matching title (case-insensitive)."""
        title_lower = title.strip().lower()
        # Exact match first
        exact = self._df[self._df["title"].str.lower() == title_lower]
        if not exact.empty:
            return int(exact.index[0])
        # Partial match
        partial = self._df[self._df["title"].str.lower().str.contains(
            title_lower, regex=False
        )]
        if not partial.empty:
            return int(partial.index[0])
        raise ValueError(f"Book not found: '{title}'")

    def _resolve_isbn(self, isbn: str) -> int:
        matches = self._df[self._df["isbn"] == isbn.strip()]
        if matches.empty:
            raise ValueError(f"ISBN not found: '{isbn}'")
        return int(matches.index[0])

    def _get_recommendations(
        self,
        idx: int,
        top_n: int,
        filter_genre: Optional[str] = None,
        filter_author: Optional[str] = None,
    ) -> List[RecommendedBook]:
        query_vec = self._feature_matrix[idx]
        scores = cosine_similarity(query_vec, self._feature_matrix).flatten()

        # Rank all books by score
        ranked = np.argsort(scores)[::-1]

        results: List[RecommendedBook] = []
        for i in ranked:
            if i == idx:           # skip the query book itself
                continue
            score = scores[i]
            if score < self.config.min_similarity:
                break

            row = self._df.iloc[i]

            # Optional filters
            if filter_genre and row.get("genre", "") != filter_genre:
                continue
            if filter_author and row.get("author", "") == filter_author:
                continue

            results.append(self._build_result(i, score))
            if len(results) >= top_n:
                break

        return results

    def _build_result(self, idx: int, score: float) -> RecommendedBook:
        row = self._df.iloc[idx]
        return RecommendedBook(
            isbn=str(row.get("isbn", "")),
            title=str(row.get("title", "")),
            author=str(row.get("author", "")),
            genre=str(row.get("genre", "")),
            subgenre=str(row.get("subgenre", "")),
            description=str(row.get("description", "")),
            avg_rating=float(row.get("avg_rating", 0) or 0),
            num_ratings=int(row.get("num_ratings", 0) or 0),
            year=int(row["year"]) if pd.notna(row.get("year")) else None,
            pages=int(row["pages"]) if pd.notna(row.get("pages")) else None,
            language=str(row.get("language", "English")),
            similarity_score=float(score),
        )

    def _row_to_dict(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "isbn": str(row.get("isbn", "")),
            "title": str(row.get("title", "")),
            "author": str(row.get("author", "")),
            "genre": str(row.get("genre", "")),
            "subgenre": str(row.get("subgenre", "")),
            "description": str(row.get("description", "")),
            "avg_rating": float(row.get("avg_rating", 0) or 0),
            "num_ratings": int(row.get("num_ratings", 0) or 0),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "pages": int(row["pages"]) if pd.notna(row.get("pages")) else None,
            "language": str(row.get("language", "English")),
        }
