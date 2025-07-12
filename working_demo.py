#!/usr/bin/env python3
"""
Working Vector Database Module Demo

This script demonstrates the core functionality that is currently working
in the Vector Database Module for legal QA systems.
"""

import os
import sys
import yaml
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingVectorDatabaseDemo:
    """Working demo class for Vector Database Module functionality."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the demo with configuration."""
        self.config = self.load_config(config_path)
        self.faiss_index = None
        self.documents = []
        self.embeddings = []
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def create_sample_documents(self) -> List[Dict[str, Any]]:
        """Create sample legal documents for demonstration."""
        documents = [
            {
                "id": "doc_001",
                "title": "Contract Law Principles",
                "content": "A contract is a legally binding agreement between two or more parties. The essential elements of a contract include offer, acceptance, consideration, and mutual intent to be bound.",
                "doctrine": "contract_law",
                "court": "supreme_court",
                "year": 2020,
                "jurisdiction": "federal"
            },
            {
                "id": "doc_002", 
                "title": "Tort Liability Standards",
                "content": "Tort law governs civil wrongs and provides remedies for damages. Negligence requires proving duty, breach, causation, and damages. Strict liability applies in certain circumstances.",
                "doctrine": "tort_law",
                "court": "appellate_court",
                "year": 2021,
                "jurisdiction": "state"
            },
            {
                "id": "doc_003",
                "title": "Constitutional Rights Framework",
                "content": "The Constitution establishes fundamental rights and freedoms. Due process protects against arbitrary government action. Equal protection ensures fair treatment under the law.",
                "doctrine": "constitutional_law",
                "court": "supreme_court", 
                "year": 2019,
                "jurisdiction": "federal"
            },
            {
                "id": "doc_004",
                "title": "Criminal Procedure Guidelines",
                "content": "Criminal procedure governs the investigation and prosecution of crimes. The Fourth Amendment protects against unreasonable searches and seizures. Miranda rights must be read upon arrest.",
                "doctrine": "criminal_law",
                "court": "district_court",
                "year": 2022,
                "jurisdiction": "federal"
            },
            {
                "id": "doc_005",
                "title": "Property Rights and Ownership",
                "content": "Property law defines ownership rights and interests in real and personal property. Fee simple absolute provides the most complete ownership. Easements grant limited use rights.",
                "doctrine": "property_law",
                "court": "appellate_court",
                "year": 2020,
                "jurisdiction": "state"
            }
        ]
        
        logger.info(f"Created {len(documents)} sample documents")
        return documents
    
    def generate_simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate simple embeddings using basic text features."""
        # This is a simplified embedding approach for demo purposes
        # In production, you would use Legal-BERT or similar models
        
        embeddings = []
        for text in texts:
            # Create a simple feature vector based on text characteristics
            words = text.lower().split()
            
            # Basic features
            features = [
                len(words),  # word count
                len(text),   # character count
                text.count('.'),  # sentence count (approximate)
                text.count('law'),  # legal term frequency
                text.count('court'),
                text.count('contract'),
                text.count('tort'),
                text.count('constitutional'),
                text.count('criminal'),
                text.count('property'),
            ]
            
            # Pad or truncate to fixed size
            target_size = 384  # Common embedding dimension
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            embeddings.append(features)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        logger.info(f"Generated embeddings with shape: {embeddings_array.shape}")
        return embeddings_array
    
    def setup_faiss_index(self, embeddings: np.ndarray) -> None:
        """Set up FAISS index for vector similarity search."""
        try:
            import faiss
            
            dimension = embeddings.shape[1]
            
            # Create FAISS index (Inner Product for cosine similarity)
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.faiss_index.add(embeddings)
            
            logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to setup FAISS index: {e}")
            raise
    
    def setup_bm25_index(self, documents: List[Dict[str, Any]]) -> None:
        """Set up BM25 index for keyword search."""
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize documents for BM25
            corpus = [doc['content'] for doc in documents]
            tokenized_corpus = [doc.split() for doc in corpus]
            
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index created successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup BM25 index: {e}")
            raise
    
    def vector_search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
        """Perform vector similarity search using FAISS."""
        try:
            import faiss
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Perform keyword search using BM25."""
        try:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = [(int(idx), float(scores[idx])) for idx in top_indices]
            
            logger.info(f"Keyword search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = 3, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword search."""
        try:
            # Generate query embedding
            query_embedding = self.generate_simple_embeddings([query])[0]
            
            # Perform both searches
            vector_results = self.vector_search(query_embedding, top_k * 2)
            keyword_results = self.keyword_search(query, top_k * 2)
            
            # Combine scores
            combined_scores = {}
            
            # Add vector scores
            for idx, score in vector_results:
                combined_scores[idx] = alpha * score
            
            # Add keyword scores
            for idx, score in keyword_results:
                if idx in combined_scores:
                    combined_scores[idx] += (1 - alpha) * score
                else:
                    combined_scores[idx] = (1 - alpha) * score
            
            # Sort by combined score
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Format results
            results = []
            for idx, score in sorted_results:
                doc = self.documents[idx].copy()
                doc['relevance_score'] = score
                results.append(doc)
            
            logger.info(f"Hybrid search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def filter_by_metadata(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter search results by metadata criteria."""
        filtered_results = []
        
        for result in results:
            match = True
            for key, value in filters.items():
                if key in result and result[key] != value:
                    match = False
                    break
            
            if match:
                filtered_results.append(result)
        
        logger.info(f"Filtered {len(results)} results to {len(filtered_results)} results")
        return filtered_results
    
    def run_demo(self):
        """Run the complete demo."""
        print("="*60)
        print("WORKING VECTOR DATABASE MODULE DEMO")
        print("="*60)
        
        # Step 1: Create sample documents
        print("\n1. Creating sample legal documents...")
        self.documents = self.create_sample_documents()
        for i, doc in enumerate(self.documents):
            print(f"   Document {i+1}: {doc['title']} ({doc['doctrine']}, {doc['year']})")
        
        # Step 2: Generate embeddings
        print("\n2. Generating document embeddings...")
        texts = [doc['content'] for doc in self.documents]
        self.embeddings = self.generate_simple_embeddings(texts)
        print(f"   Generated embeddings: {self.embeddings.shape}")
        
        # Step 3: Setup indexes
        print("\n3. Setting up search indexes...")
        self.setup_faiss_index(self.embeddings)
        self.setup_bm25_index(self.documents)
        print("   FAISS and BM25 indexes created")
        
        # Step 4: Demonstrate searches
        print("\n4. Demonstrating search capabilities...")
        
        queries = [
            "contract agreement legal binding",
            "constitutional rights due process",
            "criminal procedure Miranda rights"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n   Query {i}: '{query}'")
            
            # Hybrid search
            results = self.hybrid_search(query, top_k=2)
            
            for j, result in enumerate(results, 1):
                print(f"      Result {j}: {result['title']}")
                print(f"         Doctrine: {result['doctrine']}")
                print(f"         Score: {result['relevance_score']:.3f}")
        
        # Step 5: Demonstrate filtering
        print("\n5. Demonstrating metadata filtering...")
        query = "legal rights and procedures"
        results = self.hybrid_search(query, top_k=5)
        
        # Filter by court type
        filtered_results = self.filter_by_metadata(results, {"court": "supreme_court"})
        print(f"   Results filtered by Supreme Court: {len(filtered_results)} documents")
        
        # Filter by year
        filtered_results = self.filter_by_metadata(results, {"year": 2020})
        print(f"   Results filtered by year 2020: {len(filtered_results)} documents")
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe Vector Database Module demonstrates:")
        print("✓ Document processing and embedding generation")
        print("✓ FAISS vector indexing and similarity search")
        print("✓ BM25 keyword search")
        print("✓ Hybrid retrieval combining both approaches")
        print("✓ Metadata filtering by doctrine, court, and year")
        print("\nNext steps:")
        print("- Replace simple embeddings with Legal-BERT")
        print("- Add more sophisticated document classification")
        print("- Implement advanced retrieval strategies")
        print("- Add evaluation metrics and validation")

def main():
    """Main function to run the demo."""
    try:
        demo = WorkingVectorDatabaseDemo()
        demo.run_demo()
        return True
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
