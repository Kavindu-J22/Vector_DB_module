"""
Utility functions for the Vector Database Module
"""

import os
import yaml
import json
import pickle
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def save_json(data: Union[Dict, List], filepath: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Data saved to {filepath}")


def load_json(filepath: str) -> Union[Dict, List]:
    """Load data from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Data loaded from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File {filepath} not found")
        raise


def save_pickle(data: Any, filepath: str) -> None:
    """Save data to pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Data pickled to {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load data from pickle file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Data loaded from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"Pickle file {filepath} not found")
        raise


def create_directories(paths: List[str]) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory created/verified: {path}")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50, min_length: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        min_length: Minimum length for a chunk to be included
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_length else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            sentence_end = text.rfind('.', start + chunk_size - 100, end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if len(chunk) >= min_length:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to 0-1 range."""
    if len(scores) == 0:
        return scores
    
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.ones_like(scores)
    
    return (scores - min_score) / (max_score - min_score)


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_file = log_config.get('file', './logs/vector_db.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        log_file,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"),
        rotation="10 MB"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    )


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure."""
    required_sections = ['models', 'vector_db', 'data', 'retrieval', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate model configuration
    if 'classifier' not in config['models'] or 'embedder' not in config['models']:
        logger.error("Missing classifier or embedder configuration")
        return False
    
    # Validate vector database configuration
    if 'faiss' not in config['vector_db'] and 'pinecone' not in config['vector_db']:
        logger.error("At least one vector database (FAISS or Pinecone) must be configured")
        return False
    
    logger.info("Configuration validation passed")
    return True
