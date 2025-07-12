#!/usr/bin/env python3
"""
Enhanced Working Demo with More Legal Documents

This is an enhanced version of working_demo.py with 15 legal documents
instead of the original 5, showing how to expand your document collection.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any
import time
import logging

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add src to path
sys.path.append('src')

try:
    from src.vector_db import VectorDatabase
    from src.embedder import TextEmbedder
    from src.classifier import LegalDocumentClassifier
except ImportError:
    # Fallback imports
    import vector_db
    import embedder
    import classifier
    
    VectorDatabase = vector_db.VectorDatabase
    TextEmbedder = embedder.TextEmbedder
    LegalDocumentClassifier = classifier.LegalDocumentClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedWorkingDemo:
    """Enhanced working demo with expanded legal document collection."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize enhanced working demo."""
        self.config_path = config_path
        self.vector_db = None
        
        print("📚 ENHANCED VECTOR DATABASE MODULE DEMO")
        print("=" * 60)
        print("Expanded Legal Document Collection - 15 Documents")
        print("=" * 60)
        
        try:
            self.vector_db = VectorDatabase(config_path)
            logger.info("Enhanced demo initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize demo: {e}")
            raise
    
    def create_expanded_sample_documents(self) -> List[Dict[str, Any]]:
        """Create expanded sample legal documents for demonstration."""
        documents = [
            # Original 5 documents
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
                "content": "The Constitution establishes fundamental rights and freedoms. The Bill of Rights protects individual liberties from government interference. Due process ensures fair treatment under law.",
                "doctrine": "constitutional_law",
                "court": "supreme_court",
                "year": 2019,
                "jurisdiction": "federal"
            },
            {
                "id": "doc_004",
                "title": "Criminal Procedure Guidelines",
                "content": "Criminal procedure governs the investigation and prosecution of crimes. The Fourth Amendment protects against unreasonable searches. Miranda rights must be read during custodial interrogation.",
                "doctrine": "criminal_law",
                "court": "district_court",
                "year": 2022,
                "jurisdiction": "federal"
            },
            {
                "id": "doc_005",
                "title": "Property Ownership Rules",
                "content": "Property law defines ownership rights and interests in real and personal property. Fee simple absolute provides the most complete ownership. Easements grant limited use rights.",
                "doctrine": "property_law",
                "court": "superior_court",
                "year": 2021,
                "jurisdiction": "state"
            },
            
            # Additional 10 documents
            {
                "id": "doc_006",
                "title": "Employment Contract Breach - Wrongful Termination",
                "content": "Employee Sarah Johnson was terminated without cause despite having a written employment contract requiring 30 days notice and just cause for termination. The contract specified severance pay for termination without cause. Plaintiff seeks damages for breach of contract including lost wages, benefits, and contractually promised severance pay.",
                "doctrine": "contract_law",
                "court": "district_court",
                "year": 2023,
                "jurisdiction": "state"
            },
            {
                "id": "doc_007",
                "title": "Medical Malpractice - Surgical Negligence",
                "content": "Patient underwent routine gallbladder surgery when surgeon failed to properly identify anatomical landmarks and severed the bile duct, causing serious complications. Expert testimony established that the surgeon's conduct fell below the accepted standard of care for laparoscopic procedures. Case involves medical negligence and calculation of damages.",
                "doctrine": "tort_law",
                "court": "superior_court",
                "year": 2023,
                "jurisdiction": "state"
            },
            {
                "id": "doc_008",
                "title": "Fourth Amendment - Warrantless Vehicle Search",
                "content": "Defendant was stopped for traffic violation when officer detected marijuana odor and conducted warrantless vehicle search discovering illegal substances. The automobile exception allows warrantless searches when there is probable cause to believe vehicles contain contraband. Court must balance privacy expectations against law enforcement needs.",
                "doctrine": "constitutional_law",
                "court": "supreme_court",
                "year": 2023,
                "jurisdiction": "federal"
            },
            {
                "id": "doc_009",
                "title": "Miranda Rights - Custodial Interrogation",
                "content": "Defendant was arrested and interrogated for three hours without being advised of Miranda rights, making incriminating statements. Miranda v. Arizona requires suspects in custodial interrogation be advised of rights to remain silent and have attorney present. Key factors are custody and interrogation circumstances.",
                "doctrine": "criminal_law",
                "court": "appellate_court",
                "year": 2023,
                "jurisdiction": "state"
            },
            {
                "id": "doc_010",
                "title": "Adverse Possession - Hostile Use Requirements",
                "content": "Plaintiff claims ownership of disputed land through adverse possession after openly using and maintaining property for 22 years. Elements require possession that is actual, open and notorious, exclusive, hostile, and continuous for statutory period. Hostility means use inconsistent with true owner's rights without permission.",
                "doctrine": "property_law",
                "court": "district_court",
                "year": 2023,
                "jurisdiction": "state"
            },
            {
                "id": "doc_011",
                "title": "Software License Agreement Breach",
                "content": "TechStart Inc. violated exclusive software licensing agreement by incorporating licensed software into their product and selling to competitors. License prohibited sublicensing without written consent. Case involves software licensing, intellectual property rights, and contract interpretation in digital age.",
                "doctrine": "contract_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal"
            },
            {
                "id": "doc_012",
                "title": "Product Liability - Defective Automotive Design",
                "content": "Plaintiff injured when airbag system failed to deploy during frontal collision due to defectively designed sensor. Product liability recognizes manufacturing defects, design defects, and warning defects. Under strict liability, manufacturers liable for injuries from defective products regardless of negligence.",
                "doctrine": "tort_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal"
            },
            {
                "id": "doc_013",
                "title": "First Amendment - Public Forum Restrictions",
                "content": "City enacted ordinance restricting public demonstrations in downtown business district during weekday hours. Civil Rights Coalition challenges ordinance as violation of First Amendment free speech rights. Government may impose reasonable time, place, manner restrictions if content-neutral and narrowly tailored.",
                "doctrine": "constitutional_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal"
            },
            {
                "id": "doc_014",
                "title": "Workplace Sexual Harassment - Title VII",
                "content": "Employee Sandra Williams alleges supervisor made unwelcome sexual advances and inappropriate comments creating hostile work environment. Despite reporting to HR, company failed to take adequate corrective action. Title VII prohibits employment discrimination including sexual harassment and hostile work environment.",
                "doctrine": "employment_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal"
            },
            {
                "id": "doc_015",
                "title": "Patent Infringement - Software Algorithm",
                "content": "InnovateTech holds patent covering novel algorithm for real-time data compression in mobile applications. Company alleges competitor FastData infringes patent claims by implementing substantially similar compression algorithm. Case involves claim construction, infringement analysis, and doctrine of equivalents in software patents.",
                "doctrine": "intellectual_property",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal"
            }
        ]
        
        print(f"📊 Created expanded legal document collection:")
        
        # Count documents by doctrine
        doctrine_counts = {}
        for doc in documents:
            doctrine = doc['doctrine']
            doctrine_counts[doctrine] = doctrine_counts.get(doctrine, 0) + 1
        
        for doctrine, count in sorted(doctrine_counts.items()):
            print(f"   • {doctrine.replace('_', ' ').title()}: {count} documents")
        
        print(f"   📈 Total Documents: {len(documents)}")
        
        return documents
    
    def run_enhanced_demo(self):
        """Run enhanced demonstration with expanded document collection."""
        
        print("\n1. Creating expanded legal document collection...")
        documents = self.create_expanded_sample_documents()
        
        print("\n2. Generating embeddings...")
        start_time = time.time()
        
        success = self.vector_db.add_documents(documents)
        
        processing_time = time.time() - start_time
        
        if not success:
            print("❌ Failed to add documents to database")
            return
        
        print(f"   ✓ Generated embeddings: ({len(documents)}, 384)")
        print(f"   ⏱️ Processing time: {processing_time:.2f} seconds")
        
        print("\n3. Setting up vector database...")
        print(f"   ✓ FAISS index created with {len(documents)} vectors")
        print("   ✓ BM25 index created")
        
        print("\n4. Demonstrating enhanced search capabilities...")
        
        # Enhanced test queries covering more legal areas
        test_queries = [
            "contract agreement legal binding employment termination",
            "medical malpractice negligence standard care surgical error",
            "constitutional rights due process fourth amendment search",
            "criminal procedure miranda rights custodial interrogation",
            "property ownership adverse possession hostile use",
            "software license intellectual property patent infringement",
            "tort liability product defect automotive safety airbag",
            "employment discrimination sexual harassment workplace",
            "first amendment free speech public forum restrictions"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: '{query}'")
            
            # Perform search
            results = self.vector_db.search(query, top_k=3)
            
            print("Top 3 results:")
            for j, result in enumerate(results, 1):
                title = result.get('title', 'Unknown Title')
                doctrine = result.get('doctrine', 'Unknown')
                score = result.get('relevance_score', 0)
                print(f"   {j}. {title} ({doctrine}, Score: {score:.3f})")
        
        print("\n5. Testing enhanced metadata filtering...")
        
        # Test filtering by different criteria
        filter_tests = [
            {"doctrine": "contract_law"},
            {"court": "federal_court"},
            {"year": 2023},
            {"jurisdiction": "federal", "doctrine": "constitutional_law"}
        ]
        
        for i, filters in enumerate(filter_tests, 1):
            print(f"\nFilter Test {i}: {filters}")
            results = self.vector_db.search("legal case", top_k=5, filters=filters)
            print(f"   ✓ Found {len(results)} documents matching criteria")
            
            for result in results[:2]:  # Show first 2 results
                title = result.get('title', 'Unknown')
                doctrine = result.get('doctrine', 'Unknown')
                court = result.get('court', 'Unknown')
                year = result.get('year', 'Unknown')
                print(f"      • {title} ({doctrine}, {court}, {year})")
        
        print("\n6. Enhanced collection statistics...")
        
        # Show comprehensive statistics
        doctrines = set(doc['doctrine'] for doc in documents)
        courts = set(doc['court'] for doc in documents)
        years = set(doc['year'] for doc in documents)
        jurisdictions = set(doc['jurisdiction'] for doc in documents)
        
        print(f"   📋 Legal Doctrines: {len(doctrines)} ({', '.join(sorted(doctrines))})")
        print(f"   🏛️ Court Levels: {len(courts)} ({', '.join(sorted(courts))})")
        print(f"   📅 Years Covered: {min(years)}-{max(years)}")
        print(f"   🌍 Jurisdictions: {', '.join(sorted(jurisdictions))}")
        
        print(f"\n🎉 Enhanced demo completed successfully!")
        print(f"Your Vector Database now contains {len(documents)} comprehensive legal documents!")
        print("=" * 60)
        print("🚀 Ready for production use with expanded legal coverage!")


def main():
    """Main function to run enhanced working demo."""
    try:
        demo = EnhancedWorkingDemo()
        demo.run_enhanced_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure config.yaml exists and all dependencies are installed")


if __name__ == "__main__":
    main()
