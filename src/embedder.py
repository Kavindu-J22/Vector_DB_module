"""
Document Embedding Module for Legal Documents

This module implements text embedding using Legal-BERT or Sentence-BERT
to convert legal text chunks into dense vector representations.
"""

import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import pickle
from tqdm import tqdm
from loguru import logger

from .utils import load_config, chunk_text, save_pickle, load_pickle


@dataclass
class EmbeddingResult:
    """Result of document embedding."""
    embeddings: np.ndarray
    chunks: List[str]
    metadata: Dict[str, Any]
    model_info: Dict[str, str]


class DocumentEmbedder:
    """
    Document embedding system using Legal-BERT or Sentence-BERT.
    
    Converts legal text chunks into dense vector representations suitable
    for semantic similarity search and retrieval.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the embedder with configuration."""
        self.config = load_config(config_path)
        self.embedder_config = self.config['models']['embedder']
        self.data_config = self.config['data']
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self.model_name = self.embedder_config['model_name']
        self.max_length = self.embedder_config['max_length']
        self.batch_size = self.embedder_config['batch_size']
        
        # Embedding cache
        self.embedding_cache = {}
        
        logger.info("DocumentEmbedder initialized")
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            # Try to load as SentenceTransformer first (recommended for embeddings)
            if 'sentence-transformers' in self.model_name or 'all-' in self.model_name:
                self.model = SentenceTransformer(self.model_name)
                self.model_type = 'sentence_transformer'
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            else:
                # Load as regular transformer model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model_type = 'transformer'
                logger.info(f"Loaded Transformer model: {self.model_name}")
                
                # Set model to evaluation mode
                self.model.eval()
        
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            # Fallback to a reliable model
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Falling back to {fallback_model}")
            self.model = SentenceTransformer(fallback_model)
            self.model_type = 'sentence_transformer'
            self.model_name = fallback_model
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _embed_with_transformer(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using transformer model."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize embeddings
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def _embed_with_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using SentenceTransformer model."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def embed_texts(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            use_cache: Whether to use embedding cache
            
        Returns:
            numpy array of embeddings
        """
        if self.model is None:
            self._initialize_model()
        
        # Check cache if enabled
        if use_cache:
            cache_key = hash(tuple(texts))
            if cache_key in self.embedding_cache:
                logger.info("Using cached embeddings")
                return self.embedding_cache[cache_key]
        
        # Generate embeddings based on model type
        if self.model_type == 'sentence_transformer':
            embeddings = self._embed_with_sentence_transformer(texts)
        else:
            embeddings = self._embed_with_transformer(texts)
        
        # Cache embeddings if enabled
        if use_cache:
            self.embedding_cache[cache_key] = embeddings
        
        logger.info(f"Generated embeddings for {len(texts)} texts, shape: {embeddings.shape}")
        return embeddings
    
    def embed_document(self, text: str, chunk_document: bool = True) -> EmbeddingResult:
        """
        Embed a single document, optionally chunking it first.
        
        Args:
            text: Document text to embed
            chunk_document: Whether to chunk the document before embedding
            
        Returns:
            EmbeddingResult containing embeddings, chunks, and metadata
        """
        if chunk_document:
            # Chunk the document
            chunks = chunk_text(
                text,
                chunk_size=self.data_config['chunk_size'],
                overlap=self.data_config['chunk_overlap'],
                min_length=self.data_config['min_chunk_length']
            )
            
            if not chunks:
                # If no valid chunks, use the original text
                chunks = [text]
        else:
            chunks = [text]
        
        # Generate embeddings
        embeddings = self.embed_texts(chunks)
        
        # Prepare metadata
        metadata = {
            'original_length': len(text),
            'num_chunks': len(chunks),
            'chunk_sizes': [len(chunk) for chunk in chunks],
            'embedding_dimension': embeddings.shape[1] if embeddings.size > 0 else 0,
            'chunked': chunk_document
        }
        
        # Prepare model info
        model_info = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'max_length': self.max_length
        }
        
        return EmbeddingResult(
            embeddings=embeddings,
            chunks=chunks,
            metadata=metadata,
            model_info=model_info
        )
    
    def embed_documents(self, documents: List[Dict[str, Any]], chunk_documents: bool = True) -> List[EmbeddingResult]:
        """
        Embed multiple documents.
        
        Args:
            documents: List of document dictionaries with 'text' field
            chunk_documents: Whether to chunk documents before embedding
            
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        
        for i, doc in enumerate(tqdm(documents, desc="Embedding documents")):
            text = doc.get('text', '')
            if not text:
                logger.warning(f"Empty text in document {i}")
                continue
            
            try:
                result = self.embed_document(text, chunk_document=chunk_documents)
                
                # Add document metadata to result
                if 'metadata' in doc:
                    result.metadata.update(doc['metadata'])
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error embedding document {i}: {e}")
                continue
        
        logger.info(f"Successfully embedded {len(results)} documents")
        return results
    
    def save_embeddings(self, embedding_results: List[EmbeddingResult], filepath: str):
        """Save embedding results to file."""
        save_data = {
            'embeddings': [result.embeddings for result in embedding_results],
            'chunks': [result.chunks for result in embedding_results],
            'metadata': [result.metadata for result in embedding_results],
            'model_info': embedding_results[0].model_info if embedding_results else {}
        }
        
        save_pickle(save_data, filepath)
        logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> List[EmbeddingResult]:
        """Load embedding results from file."""
        try:
            save_data = load_pickle(filepath)
            
            results = []
            for i in range(len(save_data['embeddings'])):
                result = EmbeddingResult(
                    embeddings=save_data['embeddings'][i],
                    chunks=save_data['chunks'][i],
                    metadata=save_data['metadata'][i],
                    model_info=save_data['model_info']
                )
                results.append(result)
            
            logger.info(f"Loaded {len(results)} embedding results from {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading embeddings from {filepath}: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the current model."""
        if self.model is None:
            self._initialize_model()
        
        # Test with a small text to get dimension
        test_embedding = self.embed_texts(["test text"])
        return test_embedding.shape[1]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Ensure embeddings are normalized
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def find_most_similar_chunks(self, query_embedding: np.ndarray, 
                                document_embeddings: np.ndarray, 
                                chunks: List[str], 
                                top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar chunks to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Matrix of document chunk embeddings
            chunks: List of text chunks corresponding to embeddings
            top_k: Number of top similar chunks to return
            
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        # Compute similarities
        similarities = np.dot(document_embeddings, query_embedding.reshape(-1, 1)).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return chunks with scores
        results = []
        for idx in top_indices:
            results.append((chunks[idx], similarities[idx]))
        
        return results
