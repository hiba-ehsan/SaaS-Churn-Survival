"""
Thingsty SaaS Analytics — Core ML Pipeline Package
==================================================
A production-grade customer churn prediction and analytics system.
"""

__version__ = "2.0.0"
__author__ = "Growth Analytics Team"

from .config import Config
from .loader import DataLoader
from .feature_engineering import FeatureEngineer
from .models import ModelTrainer

__all__ = ['Config', 'DataLoader', 'FeatureEngineer', 'ModelTrainer']
