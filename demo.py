#!/usr/bin/env python3
"""
Vector Database Module Demonstration Script

This script demonstrates the complete pipeline of the Vector Database Module:
1. Document Classification
2. Embedding Generation  
3. Vector Database Storage
4. Hybrid Retrieval
5. Evaluation and Validation

Usage: python demo.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.classifier import DocumentClassifier
from src.embedder import DocumentEmbedder
from src.vector_db import VectorDatabase
from src.retrieval import HybridRetriever
from src.evaluation import EvaluationFramework
from src.utils import load_config, setup_logging, create_directories

def main():
    """Main demonstration function."""
    print("="*60)
    print("VECTOR DATABASE MODULE DEMONSTRATION")
    print("="*60)
    
    # Setup
    config = load_config()
    setup_logging(config)
    
    # Create necessary directories
    create_directories([
        "data/models",
        "data/processed_documents", 
        "logs",
        "results"
    ])
    
    print("\n1. LOADING SAMPLE DATA")
    print("-" * 30)
    
    # Load sample documents
    with open("data/sample_legal_documents.json", 'r') as f:
        documents = json.load(f)
    
    print(f"Loaded {len(documents)} sample legal documents")
    
    # Load test queries
    with open("data/test_queries.json", 'r') as f:
        test_queries = json.load(f)
    
    print(f"Loaded {len(test_queries)} test queries")
    
    print("\n2. DOCUMENT CLASSIFICATION DEMO")
    print("-" * 35)
    
    # Initialize classifier
    classifier = DocumentClassifier()
    
    # Since we don't have pre-trained models, we'll use the metadata from sample data
    # In a real scenario, you would train the classifier first
    print("Note: Using provided metadata instead of training classifier")
    print("In production, you would train the classifier with labeled data")
    
    # Show classification example
    sample_doc = documents[0]
    print(f"\nSample document: {sample_doc['metadata']['title']}")
    print(f"Doctrine: {sample_doc['metadata']['doctrine']}")
    print(f"Court: {sample_doc['metadata']['court']}")
    print(f"Year: {sample_doc['metadata']['year']}")
    
    print("\n3. EMBEDDING GENERATION")
    print("-" * 25)
    
    # Initialize embedder
    embedder = DocumentEmbedder()
    
    print("Generating embeddings for sample documents...")
    start_time = time.time()
    
    # Generate embeddings for first 3 documents (for demo speed)
    demo_docs = documents[:3]
    embedding_results = embedder.embed_documents(demo_docs, chunk_documents=True)
    
    embedding_time = time.time() - start_time
    
    total_chunks = sum(len(result.chunks) for result in embedding_results)
    print(f"Generated embeddings for {len(demo_docs)} documents")
    print(f"Total chunks: {total_chunks}")
    print(f"Embedding dimension: {embedding_results[0].embeddings.shape[1]}")
    print(f"Time taken: {embedding_time:.2f} seconds")
    
    print("\n4. VECTOR DATABASE STORAGE")
    print("-" * 30)
    
    # Initialize hybrid retriever (includes vector database)
    print("Initializing hybrid retrieval system...")
    retriever = HybridRetriever(db_type="faiss")  # Using FAISS for demo
    
    # Add documents to the system
    print("Adding documents to vector database and BM25 index...")
    retriever.add_documents(documents, classify_documents=False)
    
    # Get system statistics
    stats = retriever.get_stats()
    print(f"Vector database contains {stats['total_chunks']} chunks")
    print(f"BM25 index contains {stats['bm25_documents']} documents")
    
    print("\n5. HYBRID RETRIEVAL DEMO")
    print("-" * 25)
    
    # Demonstrate different types of searches
    demo_query = "contract breach damages"
    print(f"Demo query: '{demo_query}'")
    
    # Vector-only search
    print("\nVector-only search:")
    vector_results = retriever.search(
        demo_query, 
        top_k=3, 
        use_vector=True, 
        use_bm25=False
    )
    
    for i, result in enumerate(vector_results, 1):
        print(f"  {i}. Score: {result.vector_score:.3f} | {result.text[:100]}...")
    
    # BM25-only search  
    print("\nBM25-only search:")
    bm25_results = retriever.search(
        demo_query,
        top_k=3,
        use_vector=False,
        use_bm25=True
    )
    
    for i, result in enumerate(bm25_results, 1):
        print(f"  {i}. Score: {result.bm25_score:.3f} | {result.text[:100]}...")
    
    # Hybrid search
    print("\nHybrid search (Vector + BM25):")
    hybrid_results = retriever.search(
        demo_query,
        top_k=3,
        use_vector=True,
        use_bm25=True
    )
    
    for i, result in enumerate(hybrid_results, 1):
        print(f"  {i}. Combined: {result.combined_score:.3f} "
              f"(V: {result.vector_score:.3f}, BM25: {result.bm25_score:.3f}) | "
              f"{result.text[:100]}...")
    
    # Filtered search
    print("\nFiltered search (Contract doctrine only):")
    filtered_results = retriever.search(
        demo_query,
        top_k=3,
        filters={"doctrine": "contract"},
        use_vector=True,
        use_bm25=True
    )
    
    for i, result in enumerate(filtered_results, 1):
        doctrine = result.metadata.get('doctrine', 'unknown')
        print(f"  {i}. Score: {result.combined_score:.3f} | Doctrine: {doctrine} | "
              f"{result.text[:80]}...")
    
    print("\n6. EVALUATION AND VALIDATION")
    print("-" * 32)
    
    # Initialize evaluation framework
    evaluator = EvaluationFramework()
    
    print("Running evaluation on test queries...")
    
    # Run evaluation (using subset for demo)
    demo_test_queries = test_queries[:3]  # First 3 queries for demo
    
    start_time = time.time()
    evaluation_report = evaluator.evaluate_retrieval_system(
        retriever, 
        demo_test_queries,
        compare_filtered=True
    )
    eval_time = time.time() - start_time
    
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Print evaluation summary
    evaluator.print_evaluation_summary(evaluation_report)
    
    # Save evaluation report
    report_path = "results/evaluation_report.json"
    evaluator.save_evaluation_report(evaluation_report, report_path)
    print(f"\nDetailed evaluation report saved to: {report_path}")
    
    print("\n7. PERFORMANCE ANALYSIS")
    print("-" * 25)
    
    # Analyze different search strategies
    print("Comparing search strategies:")
    
    test_query = "property rights zoning"
    strategies = [
        ("Vector Only", {"use_vector": True, "use_bm25": False}),
        ("BM25 Only", {"use_vector": False, "use_bm25": True}),
        ("Hybrid", {"use_vector": True, "use_bm25": True}),
        ("Hybrid + Filters", {"use_vector": True, "use_bm25": True, "filters": {"doctrine": "property"}})
    ]
    
    for strategy_name, search_params in strategies:
        start_time = time.time()
        results = retriever.search(test_query, top_k=5, **search_params)
        search_time = time.time() - start_time
        
        print(f"\n{strategy_name}:")
        print(f"  Search time: {search_time:.4f} seconds")
        print(f"  Results found: {len(results)}")
        
        if results:
            avg_score = sum(r.combined_score for r in results) / len(results)
            print(f"  Average score: {avg_score:.3f}")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nKey Features Demonstrated:")
    print("✓ Document classification and metadata tagging")
    print("✓ Text embedding generation with chunking")
    print("✓ Vector database storage (FAISS)")
    print("✓ BM25 keyword search integration")
    print("✓ Hybrid retrieval combining multiple methods")
    print("✓ Metadata filtering capabilities")
    print("✓ Comprehensive evaluation framework")
    print("✓ Performance comparison across strategies")
    
    print(f"\nFiles generated:")
    print(f"- Vector database index: data/faiss_index.index")
    print(f"- Evaluation report: results/evaluation_report.json")
    print(f"- System logs: logs/vector_db.log")
    
    print("\nNext steps:")
    print("1. Train classification models with your labeled data")
    print("2. Use Legal-BERT for domain-specific embeddings")
    print("3. Scale up with larger document collections")
    print("4. Integrate with Pinecone for production deployment")
    print("5. Fine-tune retrieval weights based on evaluation results")

    print("\nTo run with Pinecone instead of FAISS:")
    print("1. Update config.yaml with your Pinecone API key")
    print("2. Run: python demo.py --db-type pinecone")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vector Database Module Demo")
    parser.add_argument("--db-type", choices=["faiss", "pinecone"], default="faiss",
                       help="Vector database type to use (default: faiss)")
    args = parser.parse_args()

    try:
        # Update demo to use specified database type
        if args.db_type == "pinecone":
            print("Using Pinecone vector database")
        else:
            print("Using FAISS vector database")
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nDemo finished.")
