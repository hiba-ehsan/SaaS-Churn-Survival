import logging
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import numpy as np

def setup_logging(log_file: Path = None, level: str = 'INFO') -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_dataframe(df: pd.DataFrame, expected_columns: list = None) -> bool:
    """
    Validate DataFrame structure.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names
        
    Returns:
        True if valid, False otherwise
    """
    if df is None or len(df) == 0:
        logging.warning("DataFrame is None or empty")
        return False
    
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            logging.warning(f"Missing columns: {missing_cols}")
            return False
    
    return True


def print_data_info(df: pd.DataFrame) -> None:
    """Print detailed information about a DataFrame."""
    print("\n" + "="*60)
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    print("\nColumn Information:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nStatistical Summary:")
    print(df.describe())
    print("="*60 + "\n")


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save dictionary as JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive data summary."""
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': len(df) - len(df.drop_duplicates()),
        'memory_mb': df.memory_usage().sum() / 1024**2
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list = None):
    """
    Plot confusion matrix (requires matplotlib).
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names or ['No', 'Yes'],
                   yticklabels=class_names or ['No', 'Yes'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
    except ImportError:
        logging.warning("Matplotlib/Seaborn not available for plotting")


def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """Calculate class weights for imbalanced data."""
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    
    return {
        i: weight for i, weight in enumerate(class_weights)
    }
