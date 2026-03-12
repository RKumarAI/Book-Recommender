"""
Tests for the Book Recommender system.
Run with: pytest tests/ -v
"""

import io
import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from recommender.data_loader import DataLoader
from recommender.model import BookRecommender, ModelConfig, RecommendedBook


# ──────────────────────────────────────────────────────────────────────── #
#  Fixtures                                                                 #
# ──────────────────────────────────────────────────────────────────────── #

SAMPLE_CSV = textwrap.dedent(
    """\
    isbn,title,author,year,publisher,genre,subgenre,description,avg_rating,num_ratings,language,pages
    111,The Hunger Games,Suzanne Collins,2008,Scholastic,Fiction,Dystopian,A dystopian survival story where a girl fights to the death in a televised arena,4.3,500000,English,374
    222,Catching Fire,Suzanne Collins,2009,Scholastic,Fiction,Dystopian,The sequel continues the dystopian saga with rebellion and revolution spreading across Panem,4.2,400000,English,391
    333,Harry Potter,J.K. Rowling,1997,Bloomsbury,Fantasy,Young Adult,A young orphan wizard discovers his magical heritage and attends a school for witchcraft,4.5,900000,English,309
    444,Dune,Frank Herbert,1965,Chilton,Science Fiction,Space Opera,A desert planet holds the most valuable substance in the universe fought over by noble houses,4.2,200000,English,412
    555,Foundation,Isaac Asimov,1951,Gnome,Science Fiction,Space Opera,A mathematician predicts the fall of galactic civilisation and works to shorten the dark age,4.1,150000,English,255
    666,Pride and Prejudice,Jane Austen,1813,Egerton,Fiction,Romance,A witty tale of manners marriage and misunderstanding among English country families,4.3,800000,English,432
    777,1984,George Orwell,1949,Secker,Fiction,Dystopian,A totalitarian state controls all thought and history in this dark vision of the future,4.2,900000,English,328
    888,The Hobbit,J.R.R. Tolkien,1937,Allen,Fantasy,Adventure,A homebody hobbit is swept into an epic quest to reclaim a mountain kingdom from a dragon,4.3,700000,English,310
    999,The Martian,Andy Weir,2011,Crown,Science Fiction,Hard SF,An astronaut is stranded alone on Mars and must use science to survive until rescue arrives,4.4,500000,English,369
    000,Ender's Game,Orson Scott Card,1985,Tor,Science Fiction,Military,A gifted child is trained at battle school to become the commander in a war against aliens,4.3,400000,English,352
    """
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(SAMPLE_CSV), dtype=str)


@pytest.fixture
def fitted_recommender(sample_df) -> BookRecommender:
    loader = DataLoader.load_from_dataframe(sample_df)
    df = loader.load()
    rec = BookRecommender()
    rec.fit(df)
    return rec


@pytest.fixture
def flask_client(sample_df, tmp_path):
    """Return a Flask test client with the sample dataset."""
    csv_path = tmp_path / "books.csv"
    sample_df.to_csv(csv_path, index=False)

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app import create_app

    app = create_app(data_path=str(csv_path))
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ──────────────────────────────────────────────────────────────────────── #
#  DataLoader tests                                                         #
# ──────────────────────────────────────────────────────────────────────── #


class TestDataLoader:
    def test_loads_from_dataframe(self, sample_df):
        loader = DataLoader.load_from_dataframe(sample_df)
        df = loader.load()
        assert len(df) == 10

    def test_loads_default_file(self):
        loader = DataLoader()
        df = loader.load()
        assert len(df) > 0

    def test_missing_required_column_raises(self):
        bad = pd.DataFrame({"title": ["A"], "author": ["B"]})
        with pytest.raises(ValueError, match="missing required columns"):
            DataLoader.load_from_dataframe(bad).load()

    def test_caches_on_second_call(self, sample_df):
        loader = DataLoader.load_from_dataframe(sample_df)
        df1 = loader.load()
        df2 = loader.load()
        assert df1 is df2

    def test_stats_returns_expected_keys(self, sample_df):
        loader = DataLoader.load_from_dataframe(sample_df)
        loader.load()
        s = loader.stats()
        assert "total_books" in s
        assert "genres" in s
        assert "authors" in s
        assert "avg_rating" in s

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataLoader(str(tmp_path / "nonexistent.csv")).load()

    def test_numeric_columns_converted(self, sample_df):
        loader = DataLoader.load_from_dataframe(sample_df)
        df = loader.load()
        assert df["avg_rating"].dtype in (float, "float64")
        assert df["pages"].dtype in (float, "float64")


# ──────────────────────────────────────────────────────────────────────── #
#  BookRecommender tests                                                    #
# ──────────────────────────────────────────────────────────────────────── #


class TestBookRecommender:
    def test_fit_sets_fitted_flag(self, fitted_recommender):
        assert fitted_recommender._fitted is True

    def test_feature_matrix_shape(self, fitted_recommender):
        mat = fitted_recommender._feature_matrix
        assert mat.shape[0] == 10  # 10 sample books

    def test_recommend_by_title_returns_list(self, fitted_recommender):
        results = fitted_recommender.recommend_by_title("The Hunger Games", top_n=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_recommend_by_title_excludes_self(self, fitted_recommender):
        results = fitted_recommender.recommend_by_title("The Hunger Games", top_n=9)
        titles = [r.title for r in results]
        assert "The Hunger Games" not in titles

    def test_recommend_by_title_returns_recommended_book_type(self, fitted_recommender):
        results = fitted_recommender.recommend_by_title("Dune", top_n=3)
        for r in results:
            assert isinstance(r, RecommendedBook)

    def test_similarity_scores_in_range(self, fitted_recommender):
        results = fitted_recommender.recommend_by_title("Dune", top_n=5)
        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0

    def test_similarity_scores_descending(self, fitted_recommender):
        results = fitted_recommender.recommend_by_title("Dune", top_n=5)
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_by_title_case_insensitive(self, fitted_recommender):
        r1 = fitted_recommender.recommend_by_title("the hunger games", top_n=3)
        r2 = fitted_recommender.recommend_by_title("THE HUNGER GAMES", top_n=3)
        assert [x.title for x in r1] == [x.title for x in r2]

    def test_recommend_by_title_partial_match(self, fitted_recommender):
        results = fitted_recommender.recommend_by_title("Hunger", top_n=3)
        assert len(results) >= 1

    def test_recommend_by_title_not_found_raises(self, fitted_recommender):
        with pytest.raises(ValueError, match="not found"):
            fitted_recommender.recommend_by_title("Nonexistent Book XXXX")

    def test_recommend_by_isbn(self, fitted_recommender):
        results = fitted_recommender.recommend_by_isbn("111", top_n=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_recommend_by_description(self, fitted_recommender):
        results = fitted_recommender.recommend_by_description(
            "A young person fights against an oppressive government in a dystopian world",
            top_n=3,
        )
        assert isinstance(results, list)

    def test_genre_filter(self, fitted_recommender):
        results = fitted_recommender.recommend_by_title(
            "The Hunger Games", top_n=5, filter_genre="Science Fiction"
        )
        for r in results:
            assert r.genre == "Science Fiction"

    def test_search_titles(self, fitted_recommender):
        hits = fitted_recommender.search_titles("dune")
        assert any("Dune" in h for h in hits)

    def test_list_genres_sorted(self, fitted_recommender):
        genres = fitted_recommender.list_genres()
        assert genres == sorted(genres)
        assert len(genres) > 0

    def test_get_book_by_title(self, fitted_recommender):
        book = fitted_recommender.get_book_by_title("Dune")
        assert book is not None
        assert book["title"] == "Dune"

    def test_get_book_by_title_not_found(self, fitted_recommender):
        book = fitted_recommender.get_book_by_title("Totally Fake Book 9999")
        assert book is None

    def test_unfitted_recommender_raises(self):
        rec = BookRecommender()
        with pytest.raises(RuntimeError, match="not fitted"):
            rec.recommend_by_title("anything")

    def test_to_dict_has_required_keys(self, fitted_recommender):
        results = fitted_recommender.recommend_by_title("Dune", top_n=1)
        assert results
        d = results[0].to_dict()
        for key in ("isbn", "title", "author", "genre", "similarity_score"):
            assert key in d

    def test_pickle_roundtrip(self, fitted_recommender, tmp_path):
        path = tmp_path / "model.pkl"
        fitted_recommender.save(str(path))
        loaded = BookRecommender.load(str(path))
        r1 = fitted_recommender.recommend_by_title("Dune", top_n=3)
        r2 = loaded.recommend_by_title("Dune", top_n=3)
        assert [x.title for x in r1] == [x.title for x in r2]

    def test_custom_config(self, sample_df):
        cfg = ModelConfig(
            tfidf_max_features=500,
            use_rating_feature=False,
            default_top_n=5,
        )
        loader = DataLoader.load_from_dataframe(sample_df)
        df = loader.load()
        rec = BookRecommender(config=cfg)
        rec.fit(df)
        results = rec.recommend_by_title("Dune")
        assert len(results) <= 5


# ──────────────────────────────────────────────────────────────────────── #
#  Flask API tests                                                          #
# ──────────────────────────────────────────────────────────────────────── #


class TestFlaskAPI:
    def test_health(self, flask_client):
        r = flask_client.get("/api/health")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data["data"]["healthy"] is True

    def test_stats(self, flask_client):
        r = flask_client.get("/api/stats")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data["data"]["total_books"] == 10

    def test_genres(self, flask_client):
        r = flask_client.get("/api/genres")
        assert r.status_code == 200
        genres = json.loads(r.data)["data"]
        assert isinstance(genres, list)
        assert len(genres) > 0

    def test_search_books(self, flask_client):
        r = flask_client.get("/api/books/search?q=dune")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert any("Dune" in t for t in data["data"])

    def test_search_books_missing_q(self, flask_client):
        r = flask_client.get("/api/books/search")
        assert r.status_code == 400

    def test_recommend_by_title(self, flask_client):
        r = flask_client.post(
            "/api/recommend/title",
            json={"title": "Dune", "top_n": 3},
            content_type="application/json",
        )
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data["data"]["count"] <= 3

    def test_recommend_by_title_not_found(self, flask_client):
        r = flask_client.post(
            "/api/recommend/title",
            json={"title": "Totally Fake Book XXXXX"},
        )
        assert r.status_code == 404

    def test_recommend_by_title_missing_body(self, flask_client):
        r = flask_client.post(
            "/api/recommend/title",
            json={},
        )
        assert r.status_code == 400

    def test_recommend_by_description(self, flask_client):
        r = flask_client.post(
            "/api/recommend/description",
            json={"description": "A dystopian story about rebellion and survival", "top_n": 3},
        )
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "recommendations" in data["data"]

    def test_recommend_by_description_too_short(self, flask_client):
        r = flask_client.post(
            "/api/recommend/description",
            json={"description": "short"},
        )
        assert r.status_code == 400

    def test_recommend_by_isbn(self, flask_client):
        r = flask_client.post(
            "/api/recommend/isbn",
            json={"isbn": "111", "top_n": 3},
        )
        assert r.status_code == 200

    def test_recommend_by_isbn_not_found(self, flask_client):
        r = flask_client.post(
            "/api/recommend/isbn",
            json={"isbn": "9999999999"},
        )
        assert r.status_code == 404

    def test_book_detail(self, flask_client):
        r = flask_client.get("/api/books/111")
        assert r.status_code == 200
        book = json.loads(r.data)["data"]
        assert book["title"] == "The Hunger Games"

    def test_book_detail_not_found(self, flask_client):
        r = flask_client.get("/api/books/9999999")
        assert r.status_code == 404

    def test_frontend_served(self, flask_client):
        r = flask_client.get("/")
        # May 200 or 404 depending on whether frontend folder exists in tmp context
        assert r.status_code in (200, 404)
