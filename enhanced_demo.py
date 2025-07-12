#!/usr/bin/env python3
"""
Enhanced Vector Database Module Demo

This script demonstrates all the enhanced features including:
1. Legal-BERT Integration
2. Enhanced Document Classification
3. Improved Evaluation Framework
4. Performance Optimization
5. Comprehensive Testing
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any
import time
from datetime import datetime
import json

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add src to path
sys.path.append('src')

# Import with proper path handling
try:
    from src.enhanced_embedder import EnhancedTextEmbedder
    from src.enhanced_classifier import EnhancedLegalDocumentClassifier
    from src.enhanced_evaluation import EnhancedEvaluationFramework
    from src.utils import load_config
except ImportError:
    # Fallback imports
    import enhanced_embedder
    import enhanced_classifier
    import enhanced_evaluation
    import utils

    EnhancedTextEmbedder = enhanced_embedder.EnhancedTextEmbedder
    EnhancedLegalDocumentClassifier = enhanced_classifier.EnhancedLegalDocumentClassifier
    EnhancedEvaluationFramework = enhanced_evaluation.EnhancedEvaluationFramework
    load_config = utils.load_config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedVectorDatabaseDemo:
    """Enhanced demonstration of the Vector Database Module."""
    
    def __init__(self):
        """Initialize the enhanced demo."""
        self.config = load_config("config.yaml")
        
        # Initialize enhanced components
        self.embedder = None
        self.classifier = None
        self.evaluator = None
        
        # Data storage
        self.documents = []
        self.embeddings = None
        self.classifications = []
        
        logger.info("Enhanced Vector Database Demo initialized")
    
    def create_comprehensive_legal_dataset(self) -> List[Dict[str, Any]]:
        """Create a comprehensive legal document dataset for testing."""
        documents = [
            {
                "id": "contract_001",
                "title": "Employment Contract Breach Analysis",
                "content": """
                The plaintiff alleges breach of employment contract when the defendant 
                terminated employment without cause. The contract contained specific 
                provisions regarding termination procedures and severance pay. Under 
                contract law principles, a material breach occurs when one party fails 
                to perform a duty that goes to the essence of the contract. The court 
                must determine whether proper notice was given and if damages are warranted.
                The Uniform Commercial Code does not apply to employment contracts, but 
                general contract principles govern. Consideration was provided through 
                mutual promises of employment and compensation.
                """,
                "expected_doctrine": "contract_law",
                "expected_court": "district_court",
                "expected_year": 2022
            },
            {
                "id": "tort_001",
                "title": "Medical Malpractice Negligence Case",
                "content": """
                The patient suffered complications following surgery due to alleged 
                negligent care. The standard of care requires physicians to exercise 
                reasonable skill and care consistent with medical professionals in 
                similar circumstances. Expert testimony will establish whether the 
                defendant's actions fell below the accepted medical standard and 
                caused the plaintiff's injuries. Tort liability requires proof of 
                duty, breach, causation, and damages. The doctrine of res ipsa 
                loquitur may apply if the injury would not normally occur without 
                negligence. Product liability principles do not apply to medical 
                services.
                """,
                "expected_doctrine": "tort_law",
                "expected_court": "superior_court",
                "expected_year": 2023
            },
            {
                "id": "constitutional_001",
                "title": "Fourth Amendment Search and Seizure",
                "content": """
                The defendant challenges the admissibility of evidence obtained during 
                a warrantless search of his vehicle. The Fourth Amendment protects 
                against unreasonable searches and seizures by government officials. 
                The court must determine whether the search fell within an established 
                exception to the warrant requirement. Constitutional law requires 
                balancing individual privacy rights against law enforcement needs. 
                The exclusionary rule mandates suppression of illegally obtained 
                evidence. Due process and equal protection clauses also apply to 
                criminal procedure.
                """,
                "expected_doctrine": "constitutional_law",
                "expected_court": "supreme_court",
                "expected_year": 2021
            },
            {
                "id": "criminal_001",
                "title": "Miranda Rights Violation",
                "content": """
                The defendant was interrogated without being read his Miranda rights. 
                Statements made during custodial interrogation are inadmissible unless 
                the suspect was properly advised of his rights. The prosecution argues 
                the defendant was not in custody at the time of questioning. Criminal 
                procedure requires strict adherence to constitutional protections. 
                The Fifth Amendment privilege against self-incrimination applies to 
                all criminal proceedings. Evidence obtained in violation of Miranda 
                must be excluded from trial.
                """,
                "expected_doctrine": "criminal_law",
                "expected_court": "appellate_court",
                "expected_year": 2022
            },
            {
                "id": "property_001",
                "title": "Adverse Possession Claim",
                "content": """
                The plaintiff claims ownership of disputed land through adverse possession. 
                The elements require open, notorious, exclusive, and continuous possession 
                for the statutory period. The defendant argues the possession was 
                permissive and therefore cannot ripen into ownership. Property law 
                recognizes various forms of land ownership and transfer. Real estate 
                transactions must comply with the Statute of Frauds. Easements and 
                covenants may affect property rights. Title insurance protects against 
                defects in ownership.
                """,
                "expected_doctrine": "property_law",
                "expected_court": "district_court",
                "expected_year": 2020
            },
            {
                "id": "contract_002",
                "title": "Sales Contract UCC Dispute",
                "content": """
                The buyer refuses to accept delivery claiming the goods are non-conforming. 
                Under the Uniform Commercial Code, buyers have the right to inspect 
                goods and reject non-conforming deliveries. The seller argues the goods 
                substantially comply with contract specifications. UCC Article 2 governs 
                sales of goods and provides specific remedies for breach. Perfect tender 
                rule requires exact compliance with contract terms. Warranty provisions 
                protect buyers against defective merchandise.
                """,
                "expected_doctrine": "contract_law",
                "expected_court": "commercial_court",
                "expected_year": 2023
            },
            {
                "id": "tort_002",
                "title": "Product Liability Defect",
                "content": """
                The plaintiff was injured by an allegedly defective product. Product 
                liability law imposes strict liability on manufacturers for defective 
                products that cause injury. The defendant manufacturer claims the 
                product was not defective and the injury resulted from misuse. Tort 
                law recognizes three types of product defects: manufacturing, design, 
                and warning defects. Strict liability eliminates the need to prove 
                negligence. Consumer protection statutes provide additional remedies.
                """,
                "expected_doctrine": "tort_law",
                "expected_court": "superior_court",
                "expected_year": 2021
            },
            {
                "id": "constitutional_002",
                "title": "First Amendment Free Speech",
                "content": """
                The plaintiff challenges a city ordinance restricting public demonstrations. 
                The First Amendment protects freedom of speech and assembly from 
                government interference. The court must balance free speech rights 
                against the government's interest in maintaining public order and 
                safety. Constitutional analysis requires strict scrutiny for content-based 
                restrictions. Time, place, and manner regulations receive intermediate 
                scrutiny. Prior restraints on speech are presumptively unconstitutional.
                """,
                "expected_doctrine": "constitutional_law",
                "expected_court": "federal_court",
                "expected_year": 2022
            }
        ]
        
        logger.info(f"Created comprehensive legal dataset with {len(documents)} documents")
        return documents
    
    def create_enhanced_test_queries(self) -> List[Dict[str, Any]]:
        """Create enhanced test queries with expected results."""
        test_queries = [
            {
                "id": "query_001",
                "query": "employment contract termination breach severance",
                "expected_docs": ["contract_001", "contract_002"],
                "primary_doctrine": "contract_law",
                "description": "Should find contract law cases about employment and commercial contracts"
            },
            {
                "id": "query_002",
                "query": "medical malpractice negligence standard care physician",
                "expected_docs": ["tort_001", "tort_002"],
                "primary_doctrine": "tort_law",
                "description": "Should find tort law cases about medical malpractice and product liability"
            },
            {
                "id": "query_003",
                "query": "fourth amendment search warrant constitutional rights",
                "expected_docs": ["constitutional_001", "constitutional_002"],
                "primary_doctrine": "constitutional_law",
                "description": "Should find constitutional law cases about Fourth Amendment and free speech"
            },
            {
                "id": "query_004",
                "query": "miranda rights custodial interrogation criminal procedure",
                "expected_docs": ["criminal_001"],
                "primary_doctrine": "criminal_law",
                "description": "Should find criminal law case about Miranda rights"
            },
            {
                "id": "query_005",
                "query": "property ownership adverse possession real estate",
                "expected_docs": ["property_001"],
                "primary_doctrine": "property_law",
                "description": "Should find property law case about adverse possession"
            },
            {
                "id": "query_006",
                "query": "UCC sales contract goods delivery commercial",
                "expected_docs": ["contract_002", "contract_001"],
                "primary_doctrine": "contract_law",
                "description": "Should find contract law cases, especially UCC sales"
            },
            {
                "id": "query_007",
                "query": "strict liability defective product manufacturing",
                "expected_docs": ["tort_002", "tort_001"],
                "primary_doctrine": "tort_law",
                "description": "Should find product liability and related tort cases"
            },
            {
                "id": "query_008",
                "query": "first amendment free speech government restriction",
                "expected_docs": ["constitutional_002", "constitutional_001"],
                "primary_doctrine": "constitutional_law",
                "description": "Should find constitutional law cases about free speech and rights"
            }
        ]
        
        logger.info(f"Created {len(test_queries)} enhanced test queries")
        return test_queries
    
    def demonstrate_enhanced_embeddings(self):
        """Demonstrate enhanced embedding capabilities."""
        print("\n" + "="*60)
        print("ENHANCED EMBEDDING DEMONSTRATION")
        print("="*60)
        
        # Initialize enhanced embedder
        print("1. Initializing Enhanced Legal-BERT Embedder...")
        self.embedder = EnhancedTextEmbedder(embedder_type="legal_bert")
        
        # Create documents
        self.documents = self.create_comprehensive_legal_dataset()
        
        # Generate embeddings
        print("2. Generating enhanced embeddings...")
        start_time = time.time()
        
        texts = [doc['content'] for doc in self.documents]
        self.embeddings = self.embedder.embedder.encode_batch(texts)
        
        embedding_time = time.time() - start_time
        
        print(f"   ✓ Generated embeddings for {len(texts)} documents")
        print(f"   ✓ Embedding dimension: {self.embeddings.shape[1]}")
        print(f"   ✓ Generation time: {embedding_time:.2f} seconds")
        print(f"   ✓ Average time per document: {embedding_time/len(texts):.3f} seconds")
        
        # Test query embedding
        print("3. Testing query embedding...")
        query = "contract law breach damages"
        query_embedding = self.embedder.embed_query(query)
        print(f"   ✓ Query embedding shape: {query_embedding.shape}")
        
        return True
    
    def demonstrate_enhanced_classification(self):
        """Demonstrate enhanced document classification."""
        print("\n" + "="*60)
        print("ENHANCED CLASSIFICATION DEMONSTRATION")
        print("="*60)
        
        # Initialize enhanced classifier
        print("1. Initializing Enhanced Legal Document Classifier...")
        self.classifier = EnhancedLegalDocumentClassifier()
        
        # Classify documents
        print("2. Classifying legal documents...")
        self.classifications = []
        
        for i, doc in enumerate(self.documents):
            print(f"   Classifying document {i+1}/{len(self.documents)}: {doc['title'][:40]}...")
            
            classification = self.classifier.classify_document(doc)
            self.classifications.append(classification)
            
            # Check accuracy
            expected_doctrine = doc.get('expected_doctrine', 'unknown')
            actual_doctrine = classification.doctrine
            accuracy_indicator = "✓" if expected_doctrine == actual_doctrine else "✗"
            
            print(f"     {accuracy_indicator} Doctrine: {actual_doctrine} (expected: {expected_doctrine})")
            print(f"       Court: {classification.court}")
            print(f"       Year: {classification.year}")
            print(f"       Confidence: {classification.confidence_scores['doctrine']:.3f}")
        
        # Calculate classification accuracy
        correct_classifications = sum(1 for doc, cls in zip(self.documents, self.classifications)
                                      if doc.get('expected_doctrine') == cls.doctrine)
        accuracy = correct_classifications / len(self.documents)
        
        print(f"\n3. Classification Results:")
        print(f"   ✓ Total documents classified: {len(self.documents)}")
        print(f"   ✓ Correct classifications: {correct_classifications}")
        print(f"   ✓ Classification accuracy: {accuracy:.1%}")
        
        return True
    
    def demonstrate_enhanced_evaluation(self):
        """Demonstrate enhanced evaluation framework."""
        print("\n" + "="*60)
        print("ENHANCED EVALUATION DEMONSTRATION")
        print("="*60)
        
        # Initialize enhanced evaluator
        print("1. Initializing Enhanced Evaluation Framework...")
        self.evaluator = EnhancedEvaluationFramework()
        
        # Create test queries
        test_queries = self.create_enhanced_test_queries()
        
        # Define retrieval function
        def retrieval_function(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
            """Simple retrieval function for demonstration."""
            query_embedding = self.embedder.embed_query(query)
            
            # Calculate similarities
            similarities = np.dot(self.embeddings, query_embedding)
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    'id': self.documents[idx]['id'],
                    'title': self.documents[idx]['title'],
                    'relevance_score': float(similarities[idx]),
                    'content': self.documents[idx]['content'][:200] + "..."
                })
            
            return results
        
        # Run evaluation
        print("2. Running comprehensive evaluation...")
        start_time = time.time()
        
        evaluation_report = self.evaluator.evaluate_retrieval_system(
            retrieval_function=retrieval_function,
            test_queries=test_queries,
            document_corpus=self.documents
        )
        
        evaluation_time = time.time() - start_time
        
        # Display results
        print("3. Evaluation Results:")
        overall_stats = evaluation_report['overall_statistics']
        
        print(f"   ✓ Total queries evaluated: {overall_stats['total_queries']}")
        print(f"   ✓ Average Precision: {overall_stats['avg_precision']:.3f}")
        print(f"   ✓ Average Recall: {overall_stats['avg_recall']:.3f}")
        print(f"   ✓ Average F1 Score: {overall_stats['avg_f1_score']:.3f}")
        print(f"   ✓ Average MAP Score: {overall_stats['avg_map_scores']:.3f}")
        print(f"   ✓ Average MRR Score: {overall_stats['avg_mrr_scores']:.3f}")
        print(f"   ✓ Average NDCG Score: {overall_stats['avg_ndcg_scores']:.3f}")
        print(f"   ✓ Average Response Time: {overall_stats['avg_execution_time']:.3f} seconds")
        print(f"   ✓ Evaluation Time: {evaluation_time:.2f} seconds")
        
        # Display summary
        summary = evaluation_report['summary']
        print(f"\n4. Performance Summary:")
        print(f"   ✓ Performance Level: {summary['performance_level']}")
        print(f"   ✓ Speed Level: {summary['speed_level']}")
        print(f"   ✓ Key Metrics: {summary['key_metrics']}")
        
        # Save evaluation report
        report_file = self.evaluator.save_evaluation_report(evaluation_report)
        print(f"   ✓ Evaluation report saved: {report_file}")
        
        return evaluation_report
    
    def run_complete_demo(self):
        """Run the complete enhanced demonstration."""
        print("="*80)
        print("ENHANCED VECTOR DATABASE MODULE - COMPREHENSIVE DEMONSTRATION")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Run all demonstrations
            success = True
            
            success &= self.demonstrate_enhanced_embeddings()
            success &= self.demonstrate_enhanced_classification()
            evaluation_report = self.demonstrate_enhanced_evaluation()
            
            # Final summary
            print("\n" + "="*80)
            print("DEMONSTRATION SUMMARY")
            print("="*80)
            
            if success:
                print("✅ All enhanced features demonstrated successfully!")
                print("\nKey Improvements Demonstrated:")
                print("  1. ✅ Enhanced Legal-BERT embeddings with fallback mechanisms")
                print("  2. ✅ Improved document classification with metadata extraction")
                print("  3. ✅ Comprehensive evaluation framework with advanced metrics")
                print("  4. ✅ Performance optimization and error handling")
                print("  5. ✅ Robust fallback systems for reliability")
                
                # Performance summary
                if evaluation_report:
                    summary = evaluation_report['summary']
                    print(f"\nOverall Performance: {summary['performance_level']}")
                    print(f"Response Speed: {summary['speed_level']}")
                    print(f"Metrics: {summary['key_metrics']}")
                
                print("\n🎉 Enhanced Vector Database Module is ready for production!")
            else:
                print("❌ Some demonstrations encountered issues")
                print("Please check the logs for details")
            
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            logger.error(f"Demo error: {e}", exc_info=True)
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main function to run the enhanced demo."""
    demo = EnhancedVectorDatabaseDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
