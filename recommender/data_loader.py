"""
Data loading and preprocessing utilities for the Book Recommender.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading, validation, and preprocessing of book datasets.

    Supports CSV files with the following schema:
        Required: isbn, title, author, genre, description
        Optional: year, publisher, subgenre, avg_rating, num_ratings,
                  language, pages
    """

    REQUIRED_COLUMNS = {"isbn", "title", "author", "genre", "description"}
    OPTIONAL_COLUMNS = {
        "year", "publisher", "subgenre", "avg_rating",
        "num_ratings", "language", "pages",
    }
    DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "books.csv"

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else self.DEFAULT_DATA_PATH
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    #  Public interface                                                     #
    # ------------------------------------------------------------------ #

    def load(self) -> pd.DataFrame:
        """Load and return the cleaned books DataFrame."""
        if self._df is not None:
            return self._df

        logger.info("Loading dataset from %s", self.data_path)
        raw = self._read_csv()
        validated = self._validate(raw)
        self._df = self._clean(validated)
        logger.info("Loaded %d books", len(self._df))
        return self._df

    def reload(self) -> pd.DataFrame:
        """Force reload from disk."""
        self._df = None
        return self.load()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _read_csv(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        try:
            return pd.read_csv(self.data_path, dtype=str)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV: {exc}") from exc

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        # Drop rows with no title or description
        before = len(df)
        df = df.dropna(subset=["title", "description"])
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with missing title/description", dropped)
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # String columns – strip whitespace
        str_cols = ["title", "author", "genre", "description",
                    "subgenre", "publisher", "language"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").str.strip()

        # Numeric columns
        for col in ("avg_rating", "num_ratings", "year", "pages"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Normalise avg_rating to [0, 1] for feature use
        if "avg_rating" in df.columns:
            max_r = df["avg_rating"].max()
            df["rating_norm"] = df["avg_rating"] / max_r if max_r else 0.0

        # Reset index
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    #  Utility methods                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def load_from_dataframe(df: pd.DataFrame) -> "DataLoader":
        """Wrap an existing DataFrame in a DataLoader."""
        loader = DataLoader.__new__(DataLoader)
        loader.data_path = Path("in-memory")
        loader._df = df
        return loader

    def stats(self) -> dict:
        """Return basic statistics about the loaded dataset."""
        df = self.load()
        return {
            "total_books": len(df),
            "genres": df["genre"].nunique() if "genre" in df.columns else 0,
            "authors": df["author"].nunique() if "author" in df.columns else 0,
            "avg_rating": round(float(df["avg_rating"].mean()), 2)
            if "avg_rating" in df.columns
            else None,
        }
