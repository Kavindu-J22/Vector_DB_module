# Vector Database Module
"""
Legal Document Vector Database and Embedding Module

This module implements a comprehensive vector database system for legal documents
with the following key features:

1. Document Classification: BERT-based classifier for legal document categorization
2. Text Embedding: Legal-BERT/Sentence-BERT for semantic vector representations
3. Vector Storage: FAISS and Pinecone integration with metadata support
4. Hybrid Retrieval: Combined vector, keyword, and metadata-based search
5. Evaluation Framework: Comprehensive metrics for retrieval performance

Author: Vector DB Module Implementation
Date: 2025-07-12
"""

__version__ = "1.0.0"
__author__ = "Vector DB Module Team"

from .classifier import DocumentClassifier
from .embedder import DocumentEmbedder
from .vector_db import VectorDatabase
from .retrieval import HybridRetriever
from .evaluation import EvaluationFramework

__all__ = [
    "DocumentClassifier",
    "DocumentEmbedder", 
    "VectorDatabase",
    "HybridRetriever",
    "EvaluationFramework"
]
