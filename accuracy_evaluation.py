#!/usr/bin/env python3
"""
Accuracy Evaluation and Manual Testing Script

This script provides comprehensive accuracy evaluation and manual testing
capabilities for the Vector Database Module.
"""

import os
import sys
import yaml
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import time
import json
from datetime import datetime

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccuracyEvaluator:
    """Comprehensive accuracy evaluation for the Vector Database Module."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the evaluator."""
        self.config = self.load_config(config_path)
        self.test_results = {}
        
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
    
    def create_test_dataset(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Create a comprehensive test dataset with ground truth."""
        
        # Legal documents with rich metadata
        documents = [
            {
                "id": "contract_001",
                "title": "Employment Contract Breach",
                "content": "The plaintiff alleges breach of employment contract when the defendant terminated employment without cause. The contract contained specific provisions regarding termination procedures and severance pay. The court must determine whether proper notice was given and if damages are warranted.",
                "doctrine": "contract_law",
                "court": "district_court",
                "year": 2022,
                "jurisdiction": "federal",
                "case_type": "civil",
                "keywords": ["employment", "breach", "termination", "severance", "notice"]
            },
            {
                "id": "tort_001",
                "title": "Medical Malpractice Negligence",
                "content": "The patient suffered complications following surgery due to alleged negligent care. The standard of care requires physicians to exercise reasonable skill and care. Expert testimony will establish whether the defendant's actions fell below the accepted medical standard and caused the plaintiff's injuries.",
                "doctrine": "tort_law",
                "court": "superior_court",
                "year": 2023,
                "jurisdiction": "state",
                "case_type": "civil",
                "keywords": ["medical", "malpractice", "negligence", "standard", "care"]
            },
            {
                "id": "constitutional_001",
                "title": "Fourth Amendment Search and Seizure",
                "content": "The defendant challenges the admissibility of evidence obtained during a warrantless search of his vehicle. The Fourth Amendment protects against unreasonable searches and seizures. The court must determine whether the search fell within an established exception to the warrant requirement.",
                "doctrine": "constitutional_law",
                "court": "supreme_court",
                "year": 2021,
                "jurisdiction": "federal",
                "case_type": "criminal",
                "keywords": ["fourth", "amendment", "search", "seizure", "warrant"]
            },
            {
                "id": "criminal_001",
                "title": "Miranda Rights Violation",
                "content": "The defendant was interrogated without being read his Miranda rights. Statements made during custodial interrogation are inadmissible unless the suspect was properly advised of his rights. The prosecution argues the defendant was not in custody at the time of questioning.",
                "doctrine": "criminal_law",
                "court": "appellate_court",
                "year": 2022,
                "jurisdiction": "state",
                "case_type": "criminal",
                "keywords": ["miranda", "rights", "interrogation", "custody", "statements"]
            },
            {
                "id": "property_001",
                "title": "Adverse Possession Claim",
                "content": "The plaintiff claims ownership of disputed land through adverse possession. The elements require open, notorious, exclusive, and continuous possession for the statutory period. The defendant argues the possession was permissive and therefore cannot ripen into ownership.",
                "doctrine": "property_law",
                "court": "district_court",
                "year": 2020,
                "jurisdiction": "state",
                "case_type": "civil",
                "keywords": ["adverse", "possession", "ownership", "statutory", "period"]
            },
            {
                "id": "contract_002",
                "title": "Sales Contract Dispute",
                "content": "The buyer refuses to accept delivery claiming the goods are non-conforming. Under the Uniform Commercial Code, buyers have the right to inspect goods and reject non-conforming deliveries. The seller argues the goods substantially comply with contract specifications.",
                "doctrine": "contract_law",
                "court": "commercial_court",
                "year": 2023,
                "jurisdiction": "state",
                "case_type": "civil",
                "keywords": ["sales", "contract", "UCC", "non-conforming", "delivery"]
            },
            {
                "id": "tort_002",
                "title": "Product Liability Defect",
                "content": "The plaintiff was injured by an allegedly defective product. Product liability law imposes strict liability on manufacturers for defective products that cause injury. The defendant manufacturer claims the product was not defective and the injury resulted from misuse.",
                "doctrine": "tort_law",
                "court": "superior_court",
                "year": 2021,
                "jurisdiction": "federal",
                "case_type": "civil",
                "keywords": ["product", "liability", "defective", "strict", "liability"]
            },
            {
                "id": "constitutional_002",
                "title": "First Amendment Free Speech",
                "content": "The plaintiff challenges a city ordinance restricting public demonstrations. The First Amendment protects freedom of speech and assembly. The court must balance free speech rights against the government's interest in maintaining public order and safety.",
                "doctrine": "constitutional_law",
                "court": "federal_court",
                "year": 2022,
                "jurisdiction": "federal",
                "case_type": "civil",
                "keywords": ["first", "amendment", "free", "speech", "assembly"]
            }
        ]
        
        # Test queries with expected relevant documents
        test_queries = [
            {
                "query": "employment contract termination breach",
                "expected_docs": ["contract_001", "contract_002"],
                "primary_doctrine": "contract_law",
                "description": "Should find contract law cases about employment and sales contracts"
            },
            {
                "query": "medical malpractice negligence standard care",
                "expected_docs": ["tort_001", "tort_002"],
                "primary_doctrine": "tort_law",
                "description": "Should find tort law cases about medical malpractice and product liability"
            },
            {
                "query": "fourth amendment search warrant",
                "expected_docs": ["constitutional_001"],
                "primary_doctrine": "constitutional_law",
                "description": "Should find constitutional law case about Fourth Amendment"
            },
            {
                "query": "miranda rights custodial interrogation",
                "expected_docs": ["criminal_001"],
                "primary_doctrine": "criminal_law",
                "description": "Should find criminal law case about Miranda rights"
            },
            {
                "query": "property ownership adverse possession",
                "expected_docs": ["property_001"],
                "primary_doctrine": "property_law",
                "description": "Should find property law case about adverse possession"
            },
            {
                "query": "constitutional rights free speech",
                "expected_docs": ["constitutional_001", "constitutional_002"],
                "primary_doctrine": "constitutional_law",
                "description": "Should find constitutional law cases about rights"
            },
            {
                "query": "contract law UCC sales",
                "expected_docs": ["contract_002", "contract_001"],
                "primary_doctrine": "contract_law",
                "description": "Should find contract law cases, especially UCC sales"
            },
            {
                "query": "strict liability defective product",
                "expected_docs": ["tort_002"],
                "primary_doctrine": "tort_law",
                "description": "Should find product liability case"
            }
        ]
        
        return documents, test_queries
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        embeddings = []
        for text in texts:
            words = text.lower().split()
            
            # Enhanced feature extraction
            features = [
                len(words),  # word count
                len(text),   # character count
                text.count('.'),  # sentence count
                text.count(','),  # clause complexity
                
                # Legal term frequencies
                text.lower().count('contract'),
                text.lower().count('tort'),
                text.lower().count('constitutional'),
                text.lower().count('criminal'),
                text.lower().count('property'),
                text.lower().count('law'),
                text.lower().count('court'),
                text.lower().count('plaintiff'),
                text.lower().count('defendant'),
                text.lower().count('breach'),
                text.lower().count('negligence'),
                text.lower().count('amendment'),
                text.lower().count('rights'),
                text.lower().count('liability'),
                text.lower().count('damages'),
                text.lower().count('evidence'),
                
                # Specific legal concepts
                text.lower().count('employment'),
                text.lower().count('medical'),
                text.lower().count('malpractice'),
                text.lower().count('search'),
                text.lower().count('seizure'),
                text.lower().count('miranda'),
                text.lower().count('possession'),
                text.lower().count('ownership'),
                text.lower().count('speech'),
                text.lower().count('assembly'),
            ]
            
            # Pad to 384 dimensions
            target_size = 384
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            embeddings.append(features)
        
        return np.array(embeddings, dtype=np.float32)
    
    def calculate_precision_recall(self, retrieved_docs: List[str], expected_docs: List[str]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        if not retrieved_docs:
            return 0.0, 0.0, 0.0
        
        retrieved_set = set(retrieved_docs)
        expected_set = set(expected_docs)
        
        true_positives = len(retrieved_set.intersection(expected_set))
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(expected_set) if expected_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def evaluate_search_accuracy(self, documents: List[Dict[str, Any]], test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate search accuracy using the test dataset."""
        from working_demo import WorkingVectorDatabaseDemo
        
        print("\n" + "="*60)
        print("ACCURACY EVALUATION")
        print("="*60)
        
        # Initialize the demo system
        demo = WorkingVectorDatabaseDemo()
        demo.documents = documents
        
        # Generate embeddings
        texts = [doc['content'] for doc in documents]
        demo.embeddings = demo.generate_simple_embeddings(texts)
        
        # Setup indexes
        demo.setup_faiss_index(demo.embeddings)
        demo.setup_bm25_index(documents)
        
        results = {
            'total_queries': len(test_queries),
            'query_results': [],
            'overall_metrics': {
                'avg_precision': 0.0,
                'avg_recall': 0.0,
                'avg_f1': 0.0
            }
        }
        
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        
        print(f"\nTesting {len(test_queries)} queries against {len(documents)} documents...")
        
        for i, test_query in enumerate(test_queries, 1):
            query = test_query['query']
            expected_docs = test_query['expected_docs']
            
            print(f"\nQuery {i}: '{query}'")
            print(f"Expected: {expected_docs}")
            
            # Perform hybrid search
            search_results = demo.hybrid_search(query, top_k=5)
            retrieved_docs = [result['id'] for result in search_results]
            
            print(f"Retrieved: {retrieved_docs[:3]}...")  # Show top 3
            
            # Calculate metrics
            precision, recall, f1 = self.calculate_precision_recall(retrieved_docs, expected_docs)
            
            print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            # Store results
            query_result = {
                'query': query,
                'expected_docs': expected_docs,
                'retrieved_docs': retrieved_docs,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'description': test_query['description']
            }
            results['query_results'].append(query_result)
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        
        # Calculate overall metrics
        results['overall_metrics']['avg_precision'] = total_precision / len(test_queries)
        results['overall_metrics']['avg_recall'] = total_recall / len(test_queries)
        results['overall_metrics']['avg_f1'] = total_f1 / len(test_queries)
        
        return results
    
    def run_manual_tests(self):
        """Run interactive manual tests."""
        from working_demo import WorkingVectorDatabaseDemo
        
        print("\n" + "="*60)
        print("MANUAL TESTING MODE")
        print("="*60)
        
        # Initialize system
        demo = WorkingVectorDatabaseDemo()
        documents, _ = self.create_test_dataset()
        demo.documents = documents
        
        # Setup
        texts = [doc['content'] for doc in documents]
        demo.embeddings = demo.generate_simple_embeddings(texts)
        demo.setup_faiss_index(demo.embeddings)
        demo.setup_bm25_index(documents)
        
        print(f"\nSystem ready with {len(documents)} legal documents")
        print("Available doctrines: contract_law, tort_law, constitutional_law, criminal_law, property_law")
        print("Available courts: district_court, superior_court, supreme_court, appellate_court, etc.")
        print("Available years: 2020-2023")
        
        while True:
            print("\n" + "-"*40)
            print("Manual Test Options:")
            print("1. Search with query")
            print("2. Search with filters")
            print("3. Show document details")
            print("4. Show system statistics")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                query = input("Enter your search query: ").strip()
                if query:
                    results = demo.hybrid_search(query, top_k=3)
                    print(f"\nTop 3 results for '{query}':")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['title']} (Score: {result['relevance_score']:.3f})")
                        print(f"   Doctrine: {result['doctrine']}, Court: {result['court']}, Year: {result['year']}")
            
            elif choice == '2':
                print("\nAvailable filters:")
                print("- doctrine: contract_law, tort_law, constitutional_law, criminal_law, property_law")
                print("- court: district_court, superior_court, supreme_court, appellate_court")
                print("- year: 2020, 2021, 2022, 2023")
                
                query = input("Enter search query: ").strip()
                doctrine = input("Filter by doctrine (or press Enter to skip): ").strip()
                court = input("Filter by court (or press Enter to skip): ").strip()
                year = input("Filter by year (or press Enter to skip): ").strip()
                
                if query:
                    results = demo.hybrid_search(query, top_k=5)
                    
                    # Apply manual filters
                    filters = {}
                    if doctrine:
                        filters['doctrine'] = doctrine
                    if court:
                        filters['court'] = court
                    if year:
                        filters['year'] = int(year)
                    
                    if filters:
                        results = demo.filter_by_metadata(results, filters)
                    
                    print(f"\nResults for '{query}' with filters {filters}:")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['title']} (Score: {result['relevance_score']:.3f})")
                        print(f"   Doctrine: {result['doctrine']}, Court: {result['court']}, Year: {result['year']}")
            
            elif choice == '3':
                doc_id = input("Enter document ID (e.g., contract_001): ").strip()
                doc = next((d for d in documents if d['id'] == doc_id), None)
                if doc:
                    print(f"\nDocument: {doc['title']}")
                    print(f"ID: {doc['id']}")
                    print(f"Doctrine: {doc['doctrine']}")
                    print(f"Court: {doc['court']}")
                    print(f"Year: {doc['year']}")
                    print(f"Content: {doc['content'][:200]}...")
                else:
                    print("Document not found")
            
            elif choice == '4':
                print(f"\nSystem Statistics:")
                print(f"Total documents: {len(documents)}")
                print(f"FAISS index size: {demo.faiss_index.ntotal if demo.faiss_index else 0}")
                print(f"Embedding dimension: {demo.embeddings.shape[1] if demo.embeddings is not None else 0}")
                print(f"BM25 corpus size: {len(demo.bm25.corpus) if hasattr(demo, 'bm25') else 0}")
            
            elif choice == '5':
                print("Exiting manual test mode...")
                break
            
            else:
                print("Invalid choice. Please enter 1-5.")
    
    def save_evaluation_report(self, results: Dict[str, Any]):
        """Save evaluation results to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEvaluation report saved to: {filename}")
    
    def run_full_evaluation(self):
        """Run complete accuracy evaluation."""
        print("="*60)
        print("VECTOR DATABASE MODULE - ACCURACY EVALUATION")
        print("="*60)
        
        # Create test dataset
        documents, test_queries = self.create_test_dataset()
        print(f"Created test dataset: {len(documents)} documents, {len(test_queries)} queries")
        
        # Run accuracy evaluation
        results = self.evaluate_search_accuracy(documents, test_queries)
        
        # Display summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Queries Tested: {results['total_queries']}")
        print(f"Average Precision: {results['overall_metrics']['avg_precision']:.3f}")
        print(f"Average Recall: {results['overall_metrics']['avg_recall']:.3f}")
        print(f"Average F1 Score: {results['overall_metrics']['avg_f1']:.3f}")
        
        # Save report
        self.save_evaluation_report(results)
        
        return results

def main():
    """Main function for accuracy evaluation."""
    evaluator = AccuracyEvaluator()
    
    print("Vector Database Module - Accuracy Evaluation and Manual Testing")
    print("1. Run Full Accuracy Evaluation")
    print("2. Run Manual Testing")
    print("3. Run Both")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        evaluator.run_full_evaluation()
    elif choice == '2':
        evaluator.run_manual_tests()
    elif choice == '3':
        evaluator.run_full_evaluation()
        evaluator.run_manual_tests()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
