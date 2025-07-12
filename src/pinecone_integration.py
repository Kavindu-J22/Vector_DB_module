"""
Pinecone Cloud Integration Module

This module provides cloud-scale vector database support using Pinecone
for production deployment of the legal document retrieval system.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
import json
from datetime import datetime
import logging
from dataclasses import dataclass

try:
    from .utils import load_config
except ImportError:
    from utils import load_config

logger = logging.getLogger(__name__)


@dataclass
class PineconeConfig:
    """Configuration for Pinecone integration."""
    api_key: str
    environment: str
    index_name: str
    dimension: int
    metric: str = "cosine"
    pod_type: str = "p1.x1"
    replicas: int = 1


class PineconeVectorDatabase:
    """Pinecone-based vector database for cloud-scale deployment."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Pinecone vector database."""
        self.config = load_config(config_path)
        self.pinecone_config = self._load_pinecone_config()
        
        # Initialize Pinecone client
        self.pinecone_client = None
        self.index = None
        
        # Try to initialize Pinecone
        self._initialize_pinecone()
        
        logger.info("Pinecone Vector Database initialized")
    
    def _load_pinecone_config(self) -> PineconeConfig:
        """Load Pinecone configuration."""
        pinecone_config = self.config.get('pinecone', {})
        
        return PineconeConfig(
            api_key=pinecone_config.get('api_key', os.getenv('PINECONE_API_KEY', '')),
            environment=pinecone_config.get('environment', os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')),
            index_name=pinecone_config.get('index_name', 'legal-documents'),
            dimension=pinecone_config.get('dimension', 384),
            metric=pinecone_config.get('metric', 'cosine'),
            pod_type=pinecone_config.get('pod_type', 'p1.x1'),
            replicas=pinecone_config.get('replicas', 1)
        )
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            import pinecone
            
            if not self.pinecone_config.api_key:
                logger.warning("Pinecone API key not provided. Using mock implementation.")
                self._use_mock_implementation()
                return
            
            # Initialize Pinecone
            pinecone.init(
                api_key=self.pinecone_config.api_key,
                environment=self.pinecone_config.environment
            )
            
            self.pinecone_client = pinecone
            
            # Create or connect to index
            self._setup_index()
            
            logger.info(f"Pinecone initialized successfully with index: {self.pinecone_config.index_name}")
            
        except ImportError:
            logger.warning("Pinecone package not installed. Using mock implementation.")
            self._use_mock_implementation()
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self._use_mock_implementation()
    
    def _setup_index(self):
        """Setup Pinecone index."""
        index_name = self.pinecone_config.index_name
        
        # Check if index exists
        if index_name not in self.pinecone_client.list_indexes():
            logger.info(f"Creating new Pinecone index: {index_name}")
            
            # Create index
            self.pinecone_client.create_index(
                name=index_name,
                dimension=self.pinecone_config.dimension,
                metric=self.pinecone_config.metric,
                pod_type=self.pinecone_config.pod_type,
                replicas=self.pinecone_config.replicas
            )
            
            # Wait for index to be ready
            time.sleep(10)
        
        # Connect to index
        self.index = self.pinecone_client.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    def _use_mock_implementation(self):
        """Use mock implementation when Pinecone is not available."""
        logger.info("Using mock Pinecone implementation for demonstration")
        self.mock_vectors = {}
        self.mock_metadata = {}
        self.is_mock = True
    
    def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> bool:
        """Upsert documents with embeddings to Pinecone."""
        try:
            if hasattr(self, 'is_mock'):
                return self._mock_upsert_documents(documents, embeddings)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                vector_data = {
                    'id': doc['id'],
                    'values': embedding.tolist(),
                    'metadata': {
                        'title': doc.get('title', ''),
                        'doctrine': doc.get('doctrine', ''),
                        'court': doc.get('court', ''),
                        'year': doc.get('year', 0),
                        'jurisdiction': doc.get('jurisdiction', ''),
                        'case_type': doc.get('case_type', ''),
                        'content_preview': doc.get('content', '')[:500]  # First 500 chars
                    }
                }
                vectors.append(vector_data)
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            
            logger.info(f"Successfully upserted {len(documents)} documents to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert documents to Pinecone: {e}")
            return False
    
    def _mock_upsert_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> bool:
        """Mock implementation of document upsert."""
        for doc, embedding in zip(documents, embeddings):
            self.mock_vectors[doc['id']] = embedding
            self.mock_metadata[doc['id']] = {
                'title': doc.get('title', ''),
                'doctrine': doc.get('doctrine', ''),
                'court': doc.get('court', ''),
                'year': doc.get('year', 0),
                'content_preview': doc.get('content', '')[:500]
            }
        
        logger.info(f"Mock upserted {len(documents)} documents")
        return True
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 10, 
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents using query embedding."""
        try:
            if hasattr(self, 'is_mock'):
                return self._mock_search(query_embedding, top_k, filters)
            
            # Prepare filter
            pinecone_filter = self._prepare_pinecone_filter(filters) if filters else None
            
            # Perform search
            search_results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            # Process results
            results = []
            for match in search_results['matches']:
                result = {
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match.get('metadata', {})
                }
                results.append(result)
            
            logger.info(f"Pinecone search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []
    
    def _mock_search(self, query_embedding: np.ndarray, 
                     top_k: int = 10, 
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Mock implementation of search."""
        if not self.mock_vectors:
            return []
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_embedding in self.mock_vectors.items():
            # Apply filters
            if filters:
                metadata = self.mock_metadata[doc_id]
                if not self._apply_filters(metadata, filters):
                    continue
            
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((doc_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for doc_id, score in similarities[:top_k]:
            results.append({
                'id': doc_id,
                'score': float(score),
                'metadata': self.mock_metadata[doc_id]
            })
        
        logger.info(f"Mock search returned {len(results)} results")
        return results
    
    def _prepare_pinecone_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare filters for Pinecone query."""
        pinecone_filter = {}
        
        for key, value in filters.items():
            if key in ['doctrine', 'court', 'jurisdiction', 'case_type']:
                pinecone_filter[key] = {"$eq": value}
            elif key == 'year':
                if isinstance(value, dict):
                    # Range filter
                    if 'min' in value and 'max' in value:
                        pinecone_filter[key] = {"$gte": value['min'], "$lte": value['max']}
                    elif 'min' in value:
                        pinecone_filter[key] = {"$gte": value['min']}
                    elif 'max' in value:
                        pinecone_filter[key] = {"$lte": value['max']}
                else:
                    # Exact match
                    pinecone_filter[key] = {"$eq": value}
        
        return pinecone_filter
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to metadata (for mock implementation)."""
        for key, value in filters.items():
            if key not in metadata:
                continue
            
            if isinstance(value, dict) and key == 'year':
                # Range filter
                year = metadata[key]
                if 'min' in value and year < value['min']:
                    return False
                if 'max' in value and year > value['max']:
                    return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            if hasattr(self, 'is_mock'):
                return {
                    'total_vector_count': len(self.mock_vectors),
                    'dimension': 384,
                    'index_fullness': 0.0,
                    'is_mock': True
                }
            
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0.0),
                'is_mock': False
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the index."""
        try:
            if hasattr(self, 'is_mock'):
                for doc_id in document_ids:
                    self.mock_vectors.pop(doc_id, None)
                    self.mock_metadata.pop(doc_id, None)
                logger.info(f"Mock deleted {len(document_ids)} documents")
                return True
            
            self.index.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def clear_index(self) -> bool:
        """Clear all vectors from the index."""
        try:
            if hasattr(self, 'is_mock'):
                self.mock_vectors.clear()
                self.mock_metadata.clear()
                logger.info("Mock index cleared")
                return True
            
            self.index.delete(delete_all=True)
            logger.info("Pinecone index cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False


class PineconeHybridRetriever:
    """Hybrid retriever combining Pinecone vector search with keyword search."""
    
    def __init__(self, pinecone_db: PineconeVectorDatabase, bm25_index=None):
        """Initialize hybrid retriever."""
        self.pinecone_db = pinecone_db
        self.bm25_index = bm25_index
        self.vector_weight = 0.7
        self.keyword_weight = 0.3
        
        logger.info("Pinecone Hybrid Retriever initialized")
    
    def search(self, query_embedding: np.ndarray, query_text: str,
               top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword approaches."""
        
        # Vector search using Pinecone
        vector_results = self.pinecone_db.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more results for reranking
            filters=filters
        )
        
        # If BM25 is available, combine with keyword search
        if self.bm25_index:
            # This would require implementing BM25 integration
            # For now, just use vector results
            pass
        
        # Rerank and return top-k results
        final_results = vector_results[:top_k]
        
        logger.info(f"Hybrid search returned {len(final_results)} results")
        return final_results
