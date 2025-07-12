#!/usr/bin/env python3
"""
Minimal Test Script for Vector Database Module

This script performs a basic test to verify that the core dependencies
are installed and the basic functionality works.
"""

import sys
import traceback
import os

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_core_imports():
    """Test if core packages can be imported."""
    print("Testing core imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import faiss
        print(f"✓ FAISS")
    except ImportError as e:
        print(f"✗ FAISS import failed: {e}")
        return False
    
    try:
        import pinecone
        print(f"✓ Pinecone")
    except ImportError as e:
        print(f"✗ Pinecone import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import yaml
        print(f"✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False
    
    try:
        from rank_bm25 import BM25Okapi
        print(f"✓ Rank BM25")
    except ImportError as e:
        print(f"✗ Rank BM25 import failed: {e}")
        return False
    
    return True

def test_core_functionality():
    """Test core functionality of key components."""
    print("\nTesting core functionality...")
    
    try:
        # Test FAISS
        import faiss
        import numpy as np
        
        # Create a simple index
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        
        # Add some random vectors
        vectors = np.random.random((10, dimension)).astype('float32')
        faiss.normalize_L2(vectors)
        index.add(vectors)
        
        # Search
        query = np.random.random((1, dimension)).astype('float32')
        faiss.normalize_L2(query)
        scores, indices = index.search(query, 3)
        
        print(f"✓ FAISS indexing and search: {len(indices[0])} results")
    except Exception as e:
        print(f"✗ FAISS test failed: {e}")
        return False
    
    try:
        # Test BM25
        from rank_bm25 import BM25Okapi
        
        corpus = [
            "Hello world",
            "This is a test document",
            "Another test document for BM25"
        ]
        
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        
        query = "test document"
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        
        print(f"✓ BM25 scoring: {len(scores)} scores computed")
    except Exception as e:
        print(f"✗ BM25 test failed: {e}")
        return False
    
    try:
        # Test basic transformers functionality
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer("Hello world", return_tensors="pt")
        
        print(f"✓ Transformers tokenization: {len(tokens['input_ids'][0])} tokens")
    except Exception as e:
        print(f"✗ Transformers test failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['models', 'vector_db', 'data', 'retrieval', 'evaluation']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False
        
        print("✓ Configuration file loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("VECTOR DATABASE MODULE - MINIMAL TEST")
    print("="*50)
    
    all_passed = True
    
    # Test core imports
    if not test_core_imports():
        all_passed = False
    
    # Test core functionality
    if not test_core_functionality():
        all_passed = False
    
    # Test configuration
    if not test_config_loading():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL CORE TESTS PASSED! System is ready.")
        print("Note: Some advanced features may require additional packages.")
        print("You can now run:")
        print("  python demo.py")
        print("  python validate_system.py")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
    
    print("="*50)
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
