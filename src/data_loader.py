"""
Data Loading & Initial Cleaning
================================
Handles CSV loading, basic cleaning, and data validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and preprocess customer churn data."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        
    def load_csv(self, filename: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load CSV file with robust error handling.
        
        Args:
            filename: Name of the CSV file
            encoding: File encoding (default: utf-8)
            
        Returns:
            Loaded DataFrame
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
        """
        Basic data cleaning.
        
        Args:
            df: Input DataFrame
            drop_duplicates: Whether to remove duplicate rows
            
        Returns:
            Cleaned DataFrame
        """
        # Strip whitespace from column names and values
        df.columns = df.columns.str.strip()
        # Only strip object (string) columns
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]
        
        initial_rows = len(df)
        
        # Handle duplicates
        if drop_duplicates:
            df = df.drop_duplicates()
            dropped = initial_rows - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} duplicate rows")
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Log missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.info(f"Missing values detected:\n{missing[missing > 0]}")
            # Drop rows where target columns are NaN
            target_cols = [col for col in df.columns if col.lower() in ['churn', 'target']]
            if target_cols:
                rows_before_target_drop = len(df)
                df = df.dropna(subset=target_cols, how='any')
                rows_dropped = rows_before_target_drop - len(df)
                logger.info(f"Dropped {rows_dropped} rows with missing target")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            df: Input DataFrame
            strategy: 'mean', 'median', 'forward_fill', 'drop'
            
        Returns:
            DataFrame with missing values handled
        """
        if df.isnull().sum().sum() == 0:
            return df
        
        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'mean':
            return df.fillna(df.mean(numeric_only=True))
        elif strategy == 'median':
            return df.fillna(df.median(numeric_only=True))
        elif strategy == 'forward_fill':
            return df.ffill().bfill()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligently convert data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with optimized dtypes
        """
        for col in df.columns:
            # Skip if already correctly typed
            if df[col].dtype == 'object':
                # Try numeric conversion
                if col.lower() in ['charges', 'monthly', 'total', 'revenue', 'salary']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
                # Try datetime conversion
                elif col.lower() in ['date', 'time', 'created', 'updated', 'last']:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the data."""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': f"{df.memory_usage().sum() / 1024**2:.2f} MB"
        }
    
    @staticmethod
    def load_train_test_split(
        data_dir: Path,
        train_file: str,
        test_file: str,
        clean: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both training and test datasets.
        
        Args:
            data_dir: Data directory path
            train_file: Training file name
            test_file: Test file name
            clean: Whether to apply cleaning
            
        Returns:
            Tuple of (train_df, test_df)
        """
        loader = DataLoader(data_dir)
        
        train_df = loader.load_csv(train_file)
        test_df = loader.load_csv(test_file)
        
        if clean:
            train_df = loader.clean_data(train_df)
            test_df = loader.clean_data(test_df)
        
        train_df = loader.convert_dtypes(train_df)
        test_df = loader.convert_dtypes(test_df)
        
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        return train_df, test_df
