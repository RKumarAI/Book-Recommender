"""
Book Recommender Engine
Content-based filtering for personalized book recommendations.
"""

from .model import BookRecommender
from .data_loader import DataLoader

__all__ = ["BookRecommender", "DataLoader"]
__version__ = "1.0.0"
