# Vector Database Module for Legal Document Retrieval

A comprehensive vector database and embedding system for legal document classification, storage, and retrieval. This module implements hybrid search combining semantic vector similarity, keyword matching (BM25), and metadata filtering for enhanced legal document discovery.

## Features

### 🔍 **Document Classification**
- BERT-based classifier for legal document categorization
- Automatic tagging by doctrine (contract, tort, property, etc.)
- Court level classification (supreme, appeal, district, etc.)
- Year extraction from legal text

### 🧠 **Advanced Embeddings**
- Support for Legal-BERT and Sentence-BERT models
- Intelligent text chunking for long documents
- Hierarchical embedding for preserving context
- Normalized vector representations for cosine similarity

### 🗄️ **Vector Database Integration**
- **FAISS**: High-performance local vector search
- **Pinecone**: Cloud-based managed vector database
- Metadata storage and filtering capabilities
- Scalable indexing for large document collections

### 🔎 **Hybrid Retrieval System**
- **Vector Search**: Semantic similarity using embeddings
- **BM25 Search**: Traditional keyword-based retrieval
- **Metadata Filtering**: Precise filtering by doctrine, court, year
- **Score Fusion**: Weighted combination of multiple retrieval methods

### 📊 **Comprehensive Evaluation**
- Recall@K, Precision@K, MRR, NDCG metrics
- Filtered vs unfiltered search comparison
- Performance benchmarking across search strategies
- Detailed evaluation reports with statistical analysis

## Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Vector_DB_module

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/models data/processed_documents logs results

# Configure the system
cp config.yaml.example config.yaml
# Edit config.yaml with your settings
```

### Pinecone Setup (Optional)
If using Pinecone vector database:
1. Sign up at [Pinecone.io](https://pinecone.io)
2. Get your API key and environment
3. Update `config.yaml` with your Pinecone credentials

## Quick Start

### 1. Run the Demo
```bash
python demo.py
```

This demonstrates the complete pipeline:
- Document classification and metadata tagging
- Embedding generation with text chunking
- Vector database storage and indexing
- Hybrid search with multiple strategies
- Evaluation and performance analysis

### 2. Basic Usage

```python
from src.retrieval import HybridRetriever
from src.evaluation import EvaluationFramework

# Initialize the retrieval system
retriever = HybridRetriever(db_type="faiss")

# Add documents
documents = [
    {
        "id": "doc_1",
        "text": "Contract law case about breach of agreement...",
        "metadata": {"doctrine": "contract", "year": 2023}
    }
]
retriever.add_documents(documents)

# Search with different strategies
results = retriever.search(
    query="contract breach damages",
    top_k=5,
    filters={"doctrine": "contract", "year": {"$gte": 2020}},
    use_vector=True,
    use_bm25=True
)

# Evaluate performance
evaluator = EvaluationFramework()
report = evaluator.evaluate_retrieval_system(retriever, test_queries)
```

## Configuration

The system is configured via `config.yaml`:

```yaml
models:
  classifier:
    model_name: "bert-base-uncased"  # or "nlpaueb/legal-bert-base-uncased"
    num_labels: 10
    max_length: 512
    
  embedder:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    max_length: 512
    batch_size: 32

vector_db:
  faiss:
    index_type: "IndexFlatIP"
    dimension: 384
    save_path: "./data/faiss_index"
    
  pinecone:
    api_key: "your-pinecone-api-key"
    environment: "us-east-1-aws"
    index_name: "legal-documents"

retrieval:
  vector_weight: 0.7
  bm25_weight: 0.3
  top_k: 10
```

## Module Components

### DocumentClassifier (`src/classifier.py`)
Classifies legal documents by doctrine, court level, and extracts years.

```python
classifier = DocumentClassifier()
result = classifier.classify_document(text)
print(f"Doctrine: {result.doctrine}")
print(f"Court: {result.court_level}")
print(f"Year: {result.year}")
```

### DocumentEmbedder (`src/embedder.py`)
Generates semantic embeddings for text chunks.

```python
embedder = DocumentEmbedder()
embedding_result = embedder.embed_document(text, chunk_document=True)
print(f"Generated {len(embedding_result.chunks)} chunks")
```

### VectorDatabase (`src/vector_db.py`)
Manages vector storage and similarity search.

```python
vector_db = VectorDatabase(db_type="faiss")
vector_db.add_documents(document_chunks)
results = vector_db.search(query_embedding, top_k=10, filters={"doctrine": "contract"})
```

### HybridRetriever (`src/retrieval.py`)
Combines vector search, BM25, and metadata filtering.

```python
retriever = HybridRetriever()
results = retriever.search(
    "contract breach damages",
    filters={"court": "supreme"},
    use_vector=True,
    use_bm25=True
)
```

### EvaluationFramework (`src/evaluation.py`)
Comprehensive evaluation with multiple metrics.

```python
evaluator = EvaluationFramework()
report = evaluator.evaluate_retrieval_system(retriever, test_queries)
evaluator.print_evaluation_summary(report)
```

## Training Custom Models

### 1. Prepare Training Data
```python
# Format: List of documents with labels
training_data = [
    {
        "text": "Contract dispute case...",
        "labels": {
            "doctrine": "contract",
            "court": "supreme"
        }
    }
]
```

### 2. Train Classifier
```python
classifier = DocumentClassifier()
doctrine_data, court_data = classifier.prepare_training_data(training_data)
classifier.train_classifier(doctrine_data, model_type="doctrine")
classifier.train_classifier(court_data, model_type="court")
```

## Evaluation Metrics

The system supports comprehensive evaluation:

- **Recall@K**: Proportion of relevant documents retrieved in top-K
- **Precision@K**: Proportion of retrieved documents that are relevant
- **MRR**: Mean Reciprocal Rank of first relevant document
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **F1@K**: Harmonic mean of precision and recall

## Performance Optimization

### Vector Database Selection
- **FAISS**: Best for local deployment, high-speed search
- **Pinecone**: Best for cloud deployment, managed infrastructure

### Embedding Models
- **Sentence-BERT**: General-purpose, fast inference
- **Legal-BERT**: Domain-specific, better legal understanding
- **Custom Fine-tuned**: Best performance for specific use cases

### Search Strategy Tuning
```python
# Adjust weights based on evaluation results
retrieval_config = {
    "vector_weight": 0.7,  # Increase for semantic similarity
    "bm25_weight": 0.3     # Increase for keyword matching
}
```

## API Reference

Detailed API documentation is available in the docstrings of each module. Key classes:

- `DocumentClassifier`: Legal document classification
- `DocumentEmbedder`: Text embedding generation
- `VectorDatabase`: Vector storage and search
- `HybridRetriever`: Multi-modal retrieval system
- `EvaluationFramework`: Performance evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `pytest tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this module in your research, please cite:

```bibtex
@software{vector_db_module,
  title={Vector Database Module for Legal Document Retrieval},
  author={Vector DB Module Team},
  year={2025},
  url={https://github.com/your-repo/vector-db-module}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `/docs`
- Review the demo script for usage examples

---

**Note**: This implementation follows the research specifications outlined in the Vector Database and Embedding Module literature review, incorporating best practices for legal document retrieval systems.
