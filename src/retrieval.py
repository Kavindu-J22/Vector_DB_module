"""
Hybrid Retrieval System

This module implements hybrid search combining vector similarity, 
BM25 keyword search, and metadata filtering for comprehensive 
legal document retrieval.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict
from loguru import logger

from .vector_db import VectorDatabase, SearchResult, DocumentChunk
from .embedder import DocumentEmbedder
from .classifier import DocumentClassifier
from .utils import load_config, normalize_scores


@dataclass
class HybridSearchResult:
    """Result from hybrid search combining multiple retrieval methods."""
    id: str
    text: str
    doc_id: str
    vector_score: float
    bm25_score: float
    combined_score: float
    metadata: Dict[str, Any]
    rank: int


class BM25Retriever:
    """BM25-based keyword retrieval system."""
    
    def __init__(self):
        """Initialize BM25 retriever."""
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.tokenized_docs = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple tokenization - can be enhanced with legal-specific preprocessing
        text = text.lower()
        # Remove punctuation except hyphens (important for legal terms)
        text = re.sub(r'[^\w\s\-]', ' ', text)
        # Split and filter short tokens
        tokens = [token for token in text.split() if len(token) > 2]
        return tokens
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add documents to BM25 index."""
        for chunk in chunks:
            self.documents.append(chunk.text)
            self.doc_ids.append(chunk.id)
            tokenized = self._tokenize(chunk.text)
            self.tokenized_docs.append(tokenized)
        
        # Build BM25 index
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
            logger.info(f"BM25 index built with {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append((self.doc_ids[idx], scores[idx]))
        
        return results


class HybridRetriever:
    """
    Hybrid retrieval system combining vector search, BM25, and metadata filtering.
    """
    
    def __init__(self, config_path: str = "config.yaml", db_type: str = "faiss"):
        """Initialize hybrid retriever."""
        self.config = load_config(config_path)
        self.retrieval_config = self.config['retrieval']
        
        # Initialize components
        self.vector_db = VectorDatabase(config_path, db_type)
        self.embedder = DocumentEmbedder(config_path)
        self.bm25_retriever = BM25Retriever()
        self.classifier = DocumentClassifier(config_path)
        
        # Retrieval weights
        self.vector_weight = self.retrieval_config['vector_weight']
        self.bm25_weight = self.retrieval_config['bm25_weight']
        
        # Document storage for BM25
        self.all_chunks = []
        
        logger.info("HybridRetriever initialized")
    
    def add_documents(self, documents: List[Dict[str, Any]], 
                     classify_documents: bool = True) -> None:
        """
        Add documents to the hybrid retrieval system.
        
        Args:
            documents: List of documents with 'text' field
            classify_documents: Whether to classify documents for metadata
        """
        logger.info(f"Adding {len(documents)} documents to hybrid retriever")
        
        # Classify documents if requested
        if classify_documents:
            try:
                self.classifier.load_trained_models()
                logger.info("Classifying documents...")
                
                for i, doc in enumerate(documents):
                    classification_result = self.classifier.classify_document(doc['text'])
                    
                    # Add classification metadata
                    if 'metadata' not in doc:
                        doc['metadata'] = {}
                    
                    doc['metadata'].update({
                        'doctrine': classification_result.doctrine,
                        'court': classification_result.court_level,
                        'year': classification_result.year,
                        'classification_confidence': classification_result.confidence_scores
                    })
                    
                logger.info("Document classification completed")
                
            except Exception as e:
                logger.warning(f"Could not classify documents: {e}")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embedding_results = self.embedder.embed_documents(documents, chunk_documents=True)
        
        # Prepare document chunks
        all_chunks = []
        chunk_id_counter = 0
        
        for doc_idx, (doc, embedding_result) in enumerate(zip(documents, embedding_results)):
            doc_id = doc.get('id', f"doc_{doc_idx}")
            base_metadata = doc.get('metadata', {})
            
            for chunk_idx, (chunk_text, embedding) in enumerate(zip(embedding_result.chunks, embedding_result.embeddings)):
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                
                # Combine metadata
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_length': len(chunk_text),
                    'embedding_model': embedding_result.model_info['model_name']
                })
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    embedding=embedding,
                    metadata=chunk_metadata,
                    doc_id=doc_id,
                    chunk_index=chunk_idx
                )
                
                all_chunks.append(chunk)
                chunk_id_counter += 1
        
        # Add to vector database
        logger.info("Adding chunks to vector database...")
        self.vector_db.add_documents(all_chunks)
        
        # Add to BM25 index
        logger.info("Building BM25 index...")
        self.bm25_retriever.add_documents(all_chunks)
        
        # Store chunks for later use
        self.all_chunks.extend(all_chunks)
        
        logger.info(f"Successfully added {len(all_chunks)} chunks from {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10, 
               filters: Optional[Dict[str, Any]] = None,
               use_vector: bool = True, use_bm25: bool = True) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining vector and BM25 retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters to apply
            use_vector: Whether to use vector search
            use_bm25: Whether to use BM25 search
            
        Returns:
            List of HybridSearchResult objects
        """
        if not use_vector and not use_bm25:
            raise ValueError("At least one of use_vector or use_bm25 must be True")
        
        results_dict = {}  # chunk_id -> result data
        
        # Vector search
        if use_vector:
            logger.info("Performing vector search...")
            query_embedding = self.embedder.embed_texts([query])[0]
            vector_results = self.vector_db.search(
                query_embedding, 
                top_k=top_k * 2,  # Get more results for combination
                filters=filters
            )
            
            # Normalize vector scores
            if vector_results:
                vector_scores = [r.score for r in vector_results]
                normalized_vector_scores = normalize_scores(np.array(vector_scores))
                
                for result, norm_score in zip(vector_results, normalized_vector_scores):
                    results_dict[result.id] = {
                        'id': result.id,
                        'text': result.text,
                        'doc_id': result.doc_id,
                        'metadata': result.metadata,
                        'vector_score': norm_score,
                        'bm25_score': 0.0
                    }
        
        # BM25 search
        if use_bm25:
            logger.info("Performing BM25 search...")
            bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
            
            if bm25_results:
                # Normalize BM25 scores
                bm25_scores = [score for _, score in bm25_results]
                normalized_bm25_scores = normalize_scores(np.array(bm25_scores))
                
                for (chunk_id, _), norm_score in zip(bm25_results, normalized_bm25_scores):
                    if chunk_id in results_dict:
                        results_dict[chunk_id]['bm25_score'] = norm_score
                    else:
                        # Find chunk data
                        chunk_data = self._get_chunk_data(chunk_id)
                        if chunk_data and self._apply_filters(chunk_data['metadata'], filters):
                            results_dict[chunk_id] = {
                                'id': chunk_id,
                                'text': chunk_data['text'],
                                'doc_id': chunk_data['doc_id'],
                                'metadata': chunk_data['metadata'],
                                'vector_score': 0.0,
                                'bm25_score': norm_score
                            }
        
        # Combine scores and rank results
        hybrid_results = []
        for chunk_id, data in results_dict.items():
            combined_score = (
                self.vector_weight * data['vector_score'] + 
                self.bm25_weight * data['bm25_score']
            )
            
            result = HybridSearchResult(
                id=data['id'],
                text=data['text'],
                doc_id=data['doc_id'],
                vector_score=data['vector_score'],
                bm25_score=data['bm25_score'],
                combined_score=combined_score,
                metadata=data['metadata'],
                rank=0  # Will be set after sorting
            )
            hybrid_results.append(result)
        
        # Sort by combined score and assign ranks
        hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
        for i, result in enumerate(hybrid_results[:top_k]):
            result.rank = i + 1
        
        logger.info(f"Hybrid search completed: {len(hybrid_results[:top_k])} results")
        return hybrid_results[:top_k]
    
    def _get_chunk_data(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk data by ID."""
        for chunk in self.all_chunks:
            if chunk.id == chunk_id:
                return {
                    'text': chunk.text,
                    'doc_id': chunk.doc_id,
                    'metadata': chunk.metadata
                }
        return None
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Apply metadata filters."""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, dict):
                # Handle range filters
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
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        vector_stats = self.vector_db.get_stats()
        
        return {
            'vector_db_stats': vector_stats,
            'total_chunks': len(self.all_chunks),
            'bm25_documents': len(self.bm25_retriever.documents),
            'vector_weight': self.vector_weight,
            'bm25_weight': self.bm25_weight
        }
