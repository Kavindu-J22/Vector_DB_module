"""
Vector Database Integration Module

This module provides integration with FAISS and Pinecone vector databases
for storing embeddings with metadata and enabling filtered hybrid search.
"""

import os
import json
import numpy as np
import faiss
import pinecone
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import pickle
from loguru import logger

from .utils import load_config, save_json, load_json, save_pickle, load_pickle


@dataclass
class DocumentChunk:
    """Represents a document chunk with embedding and metadata."""
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    doc_id: str
    chunk_index: int


@dataclass
class SearchResult:
    """Result from vector database search."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    doc_id: str


class VectorDatabaseInterface(ABC):
    """Abstract interface for vector databases."""
    
    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the database."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents from the database."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass


class FAISSVectorDB(VectorDatabaseInterface):
    """FAISS-based vector database implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize FAISS vector database."""
        self.config = config
        self.faiss_config = config['vector_db']['faiss']
        
        # Initialize FAISS index
        self.dimension = self.faiss_config['dimension']
        self.index_type = self.faiss_config['index_type']
        self.save_path = self.faiss_config['save_path']
        
        # Create index
        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexHNSWFlat":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            logger.warning(f"Unknown index type {self.index_type}, using IndexFlatIP")
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Metadata storage (FAISS doesn't store metadata directly)
        self.metadata_store = {}
        self.id_to_index = {}  # Map document IDs to FAISS indices
        self.index_to_id = {}  # Map FAISS indices to document IDs
        self.next_index = 0
        
        # Try to load existing index
        self._load_index()
        
        logger.info(f"FAISS vector database initialized with {self.index.ntotal} vectors")
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        try:
            if os.path.exists(f"{self.save_path}.index"):
                self.index = faiss.read_index(f"{self.save_path}.index")
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            if os.path.exists(f"{self.save_path}_metadata.pkl"):
                with open(f"{self.save_path}_metadata.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.metadata_store = data.get('metadata_store', {})
                    self.id_to_index = data.get('id_to_index', {})
                    self.index_to_id = data.get('index_to_id', {})
                    self.next_index = data.get('next_index', 0)
                logger.info("Loaded FAISS metadata")
        
        except Exception as e:
            logger.warning(f"Could not load existing FAISS index: {e}")
    
    def _save_index(self):
        """Save FAISS index and metadata."""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{self.save_path}.index")
            
            # Save metadata
            metadata_data = {
                'metadata_store': self.metadata_store,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'next_index': self.next_index
            }
            
            with open(f"{self.save_path}_metadata.pkl", 'wb') as f:
                pickle.dump(metadata_data, f)
            
            logger.info("FAISS index and metadata saved")
        
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to FAISS index."""
        if not chunks:
            return
        
        # Prepare embeddings
        embeddings = np.array([chunk.embedding for chunk in chunks]).astype('float32')
        
        # Normalize embeddings for cosine similarity (if using IP index)
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(embeddings)
        
        # Add to index
        start_index = self.next_index
        self.index.add(embeddings)
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            faiss_index = start_index + i
            self.metadata_store[chunk.id] = {
                'text': chunk.text,
                'metadata': chunk.metadata,
                'doc_id': chunk.doc_id,
                'chunk_index': chunk.chunk_index
            }
            self.id_to_index[chunk.id] = faiss_index
            self.index_to_id[faiss_index] = chunk.id
        
        self.next_index += len(chunks)
        
        # Save index
        self._save_index()
        
        logger.info(f"Added {len(chunks)} chunks to FAISS index")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents in FAISS index."""
        if self.index.ntotal == 0:
            return []
        
        # Prepare query embedding
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize for cosine similarity (if using IP index)
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(query_vector)
        
        # Search in FAISS
        search_k = min(top_k * 2, self.index.ntotal)  # Get more results for filtering
        scores, indices = self.index.search(query_vector, search_k)
        
        # Convert results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            chunk_id = self.index_to_id.get(idx)
            if chunk_id is None:
                continue
            
            chunk_data = self.metadata_store.get(chunk_id)
            if chunk_data is None:
                continue
            
            # Apply filters if provided
            if filters and not self._apply_filters(chunk_data['metadata'], filters):
                continue
            
            result = SearchResult(
                id=chunk_id,
                text=chunk_data['text'],
                score=float(score),
                metadata=chunk_data['metadata'],
                doc_id=chunk_data['doc_id']
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply metadata filters to a document."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, dict):
                # Handle range filters like {"$gte": 2010}
                meta_value = metadata[key]
                for op, filter_value in value.items():
                    if op == "$gte" and meta_value < filter_value:
                        return False
                    elif op == "$lte" and meta_value > filter_value:
                        return False
                    elif op == "$gt" and meta_value <= filter_value:
                        return False
                    elif op == "$lt" and meta_value >= filter_value:
                        return False
                    elif op == "$ne" and meta_value == filter_value:
                        return False
            elif isinstance(value, list):
                # Handle list filters (value must be in list)
                if metadata[key] not in value:
                    return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False
        
        return True
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents from FAISS index (rebuild required)."""
        # FAISS doesn't support deletion, so we need to rebuild
        logger.warning("FAISS doesn't support deletion. Consider rebuilding the index.")
        # TODO: Implement index rebuilding without deleted documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS database statistics."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metadata_count': len(self.metadata_store)
        }


class PineconeVectorDB(VectorDatabaseInterface):
    """Pinecone-based vector database implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Pinecone vector database."""
        self.config = config
        self.pinecone_config = config['vector_db']['pinecone']
        
        # Initialize Pinecone
        pinecone.init(
            api_key=self.pinecone_config['api_key'],
            environment=self.pinecone_config['environment']
        )
        
        self.index_name = self.pinecone_config['index_name']
        self.dimension = self.pinecone_config['dimension']
        self.metric = self.pinecone_config['metric']
        
        # Create or connect to index
        self._initialize_index()
        
        logger.info(f"Pinecone vector database initialized: {self.index_name}")
    
    def _initialize_index(self):
        """Initialize or create Pinecone index."""
        try:
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                # Create index
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = pinecone.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            raise
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to Pinecone index."""
        if not chunks:
            return
        
        # Prepare vectors for Pinecone
        vectors = []
        for chunk in chunks:
            vector_data = {
                'id': chunk.id,
                'values': chunk.embedding.tolist(),
                'metadata': {
                    'text': chunk.text,
                    'doc_id': chunk.doc_id,
                    'chunk_index': chunk.chunk_index,
                    **chunk.metadata
                }
            }
            vectors.append(vector_data)
        
        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f"Added {len(chunks)} chunks to Pinecone index")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents in Pinecone index."""
        # Prepare query
        query_vector = query_embedding.tolist()
        
        # Perform search
        search_results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filters,
            include_metadata=True
        )
        
        # Convert results
        results = []
        for match in search_results['matches']:
            metadata = match['metadata']
            result = SearchResult(
                id=match['id'],
                text=metadata.get('text', ''),
                score=match['score'],
                metadata={k: v for k, v in metadata.items() if k not in ['text', 'doc_id']},
                doc_id=metadata.get('doc_id', '')
            )
            results.append(result)
        
        return results
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents from Pinecone index."""
        # Delete by doc_id filter
        for doc_id in doc_ids:
            self.index.delete(filter={'doc_id': doc_id})
        
        logger.info(f"Deleted documents: {doc_ids}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone database statistics."""
        stats = self.index.describe_index_stats()
        return {
            'total_vectors': stats['total_vector_count'],
            'dimension': stats['dimension'],
            'index_fullness': stats.get('index_fullness', 0),
            'namespaces': stats.get('namespaces', {})
        }


class VectorDatabase:
    """Main vector database class that can use either FAISS or Pinecone."""
    
    def __init__(self, config_path: str = "config.yaml", db_type: str = "faiss"):
        """
        Initialize vector database.
        
        Args:
            config_path: Path to configuration file
            db_type: Type of database ("faiss" or "pinecone")
        """
        self.config = load_config(config_path)
        self.db_type = db_type
        
        # Initialize the appropriate database
        if db_type == "faiss":
            self.db = FAISSVectorDB(self.config)
        elif db_type == "pinecone":
            self.db = PineconeVectorDB(self.config)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        logger.info(f"VectorDatabase initialized with {db_type}")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the database."""
        self.db.add_documents(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents."""
        return self.db.search(query_embedding, top_k, filters)
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents from the database."""
        self.db.delete_documents(doc_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = self.db.get_stats()
        stats['db_type'] = self.db_type
        return stats
