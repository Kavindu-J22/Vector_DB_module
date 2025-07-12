"""
Enhanced Text Embedding Module

This module provides improved embedding generation with Legal-BERT integration,
fallback mechanisms, and enhanced accuracy for legal document processing.
"""

import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import warnings
from loguru import logger

try:
    from .utils import load_config, chunk_text
except ImportError:
    from utils import load_config, chunk_text

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str
    max_length: int
    batch_size: int
    device: str
    normalize: bool = True


class EnhancedLegalBERTEmbedder:
    """Enhanced Legal-BERT based embedder with fallback support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Enhanced Legal-BERT embedder."""
        self.config = config
        self.embedder_config = config['models']['embedder']
        
        # Model configuration
        self.model_name = self.embedder_config.get('model_name', 'nlpaueb/legal-bert-base-uncased')
        self.max_length = self.embedder_config.get('max_length', 512)
        self.batch_size = self.embedder_config.get('batch_size', 8)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer with fallback
        self._load_model_with_fallback()
        
        logger.info(f"Enhanced Legal-BERT embedder initialized: {self.model_name}")
    
    def _load_model_with_fallback(self):
        """Load the Legal-BERT model with fallback options."""
        model_options = [
            'nlpaueb/legal-bert-base-uncased',  # Primary Legal-BERT
            'bert-base-uncased',                # Standard BERT fallback
            'distilbert-base-uncased'           # Lightweight fallback
        ]
        
        for model_name in model_options:
            try:
                logger.info(f"Attempting to load model: {model_name}")
                from transformers import AutoTokenizer, AutoModel
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                
                self.model_name = model_name
                logger.info(f"Successfully loaded model: {model_name}")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        # If all models fail, use simple embeddings
        logger.error("All transformer models failed to load, using simple embeddings")
        self.model = None
        self.tokenizer = None
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts to embeddings."""
        if self.model is None:
            # Fallback to simple embeddings
            return self._simple_embeddings(texts)
        
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            try:
                batch_embeddings = self._encode_batch_internal(batch_texts)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.warning(f"Batch encoding failed, using simple embeddings: {e}")
                simple_embeddings = self._simple_embeddings(batch_texts)
                embeddings.extend(simple_embeddings)
        
        return np.array(embeddings)
    
    def _encode_batch_internal(self, texts: List[str]) -> List[np.ndarray]:
        """Internal method to encode a batch using transformer model."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use mean pooling of last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                # Mean pooling with attention mask
                attention_mask = inputs['attention_mask']
                embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            else:
                # Fallback to pooler output
                embeddings = outputs.pooler_output
        
        # Normalize if required
        if self.embedder_config.get('normalize', True):
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling with attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _simple_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate enhanced feature-based embeddings as fallback."""
        embeddings = []
        
        for text in texts:
            # Enhanced feature extraction for legal text
            words = text.lower().split()
            
            # Basic text features
            basic_features = [
                len(words),                    # word count
                len(text),                     # character count
                text.count('.'),               # sentence count
                text.count(','),               # clause complexity
                len(set(words)),               # unique word count
                np.mean([len(w) for w in words]) if words else 0,  # avg word length
            ]
            
            # Legal-specific term frequencies (enhanced set)
            legal_terms = [
                'contract', 'agreement', 'party', 'parties', 'breach', 'damages',
                'tort', 'negligence', 'liability', 'duty', 'standard', 'care',
                'constitutional', 'amendment', 'rights', 'freedom', 'due', 'process',
                'criminal', 'procedure', 'evidence', 'miranda', 'search', 'seizure',
                'property', 'ownership', 'possession', 'title', 'easement',
                'court', 'judge', 'jury', 'trial', 'appeal', 'decision',
                'law', 'legal', 'statute', 'regulation', 'rule', 'code',
                'plaintiff', 'defendant', 'petitioner', 'respondent',
                'motion', 'order', 'judgment', 'verdict', 'sentence',
                'jurisdiction', 'federal', 'state', 'supreme', 'appellate',
                'district', 'magistrate', 'circuit', 'tribunal',
                'civil', 'criminal', 'administrative', 'equity',
                'remedy', 'relief', 'injunction', 'restitution',
                'precedent', 'stare', 'decisis', 'holding', 'dicta'
            ]
            
            legal_features = [text.lower().count(term) for term in legal_terms]
            
            # Document structure features
            structure_features = [
                text.count('§'),               # section symbols
                text.count('('),               # parentheses (citations)
                text.count('['),               # brackets
                text.count('"'),               # quotations
                text.count('\n'),              # line breaks
                len([w for w in words if w.isupper()]),  # all caps words
                len([w for w in words if w.isdigit()]),  # numbers
                text.count('v.'),              # versus (case citations)
                text.count('Id.'),             # legal citations
                text.count('See'),             # legal references
            ]
            
            # Legal document type indicators
            doc_type_features = [
                text.lower().count('whereas'),
                text.lower().count('therefore'),
                text.lower().count('hereby'),
                text.lower().count('pursuant'),
                text.lower().count('notwithstanding'),
                text.lower().count('provided'),
                text.lower().count('subject to'),
                text.lower().count('in accordance'),
            ]
            
            # Combine all features
            features = basic_features + legal_features + structure_features + doc_type_features
            
            # Pad to target dimension
            target_size = 384
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            # Normalize features
            features = np.array(features, dtype=np.float32)
            if np.linalg.norm(features) > 0:
                features = features / np.linalg.norm(features)
            
            embeddings.append(features)
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        return self.encode_batch([text])[0]


class EnhancedSentenceTransformerEmbedder:
    """Enhanced Sentence-BERT based embedder with fallback."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Enhanced Sentence-BERT embedder."""
        self.config = config
        self.embedder_config = config['models']['embedder']
        
        # Model configuration
        self.model_name = self.embedder_config.get('sentence_transformer_model', 
                                                   'all-MiniLM-L6-v2')
        self.batch_size = self.embedder_config.get('batch_size', 8)
        
        # Load model with fallback
        self._load_model_with_fallback()
        
        logger.info(f"Enhanced Sentence-BERT embedder initialized: {self.model_name}")
    
    def _load_model_with_fallback(self):
        """Load the Sentence-BERT model with fallback."""
        model_options = [
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'paraphrase-MiniLM-L6-v2'
        ]
        
        for model_name in model_options:
            try:
                # Try importing sentence_transformers
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.model_name = model_name
                logger.info(f"Successfully loaded Sentence-BERT model: {model_name}")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load Sentence-BERT {model_name}: {e}")
                continue
        
        # If all models fail, use simple embeddings
        logger.error("All Sentence-BERT models failed to load, using simple embeddings")
        self.model = None
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts to embeddings."""
        if self.model is None:
            # Fallback to simple embeddings
            return self._simple_embeddings(texts)
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.embedder_config.get('normalize', True)
            )
            return embeddings
            
        except Exception as e:
            logger.warning(f"Sentence-BERT encoding failed, using simple embeddings: {e}")
            return self._simple_embeddings(texts)
    
    def _simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate simple embeddings as fallback."""
        # Use the same simple embedding logic as Legal-BERT
        embedder = EnhancedLegalBERTEmbedder(self.config)
        return np.array(embedder._simple_embeddings(texts))
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        return self.encode_batch([text])[0]


class EnhancedTextEmbedder:
    """Enhanced text embedder with improved accuracy and fallback mechanisms."""
    
    def __init__(self, config_path: str = "config.yaml", embedder_type: str = "legal_bert"):
        """
        Initialize enhanced text embedder.
        
        Args:
            config_path: Path to configuration file
            embedder_type: Type of embedder ("legal_bert" or "sentence_bert")
        """
        self.config = load_config(config_path)
        self.embedder_type = embedder_type
        
        # Initialize the appropriate embedder
        if embedder_type == "legal_bert":
            self.embedder = EnhancedLegalBERTEmbedder(self.config)
        elif embedder_type == "sentence_bert":
            self.embedder = EnhancedSentenceTransformerEmbedder(self.config)
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")
        
        # Text chunking configuration
        self.chunk_config = self.config['data']['text_chunking']
        
        logger.info(f"EnhancedTextEmbedder initialized with {embedder_type}")
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of documents with chunking.
        
        Args:
            documents: List of documents with 'text' field
            
        Returns:
            List of document chunks with embeddings
        """
        embedded_chunks = []
        
        for doc in documents:
            doc_chunks = self._embed_document(doc)
            embedded_chunks.extend(doc_chunks)
        
        logger.info(f"Embedded {len(documents)} documents into {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def _embed_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Embed a single document with chunking."""
        text = document.get('content', document.get('text', ''))
        doc_id = document.get('id', '')
        
        # Chunk the text
        chunks = chunk_text(
            text,
            chunk_size=self.chunk_config['chunk_size'],
            overlap=self.chunk_config['overlap']
        )
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.encode_batch(chunk_texts)
        
        # Create embedded chunks
        embedded_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            embedded_chunk = {
                'id': f"{doc_id}_chunk_{i}",
                'doc_id': doc_id,
                'chunk_index': i,
                'text': chunk['text'],
                'start_pos': chunk['start'],
                'end_pos': chunk['end'],
                'embedding': embedding,
                'metadata': document.get('metadata', {})
            }
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query text."""
        return self.embedder.encode_single(query)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        # Test with a sample text
        sample_embedding = self.embed_query("sample text")
        return len(sample_embedding)
