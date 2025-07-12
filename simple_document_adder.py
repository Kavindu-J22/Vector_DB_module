#!/usr/bin/env python3
"""
Simple Document Adder

This script shows you exactly how to add more legal documents
to your Vector Database Module without dependency issues.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any
import time
from datetime import datetime

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class SimpleDocumentAdder:
    """Simple system to add more legal documents."""
    
    def __init__(self):
        """Initialize document adder."""
        print("📚 SIMPLE LEGAL DOCUMENT ADDER")
        print("=" * 60)
        print("Learn how to add unlimited legal documents to your system!")
        print("=" * 60)
    
    def create_additional_legal_documents(self) -> List[Dict[str, Any]]:
        """Create additional legal documents you can add to your system."""
        
        additional_documents = [
            # Contract Law Documents
            {
                "id": "contract_new_001",
                "title": "Software License Agreement Breach",
                "content": """
                TechStart Inc. licensed proprietary software from MegaSoft Corporation under an 
                exclusive licensing agreement for use in the healthcare industry. The license 
                agreement prohibited sublicensing or distribution to third parties without written 
                consent. TechStart allegedly violated the agreement by incorporating the licensed 
                software into their product and selling it to competitors in the same market. 
                MegaSoft seeks injunctive relief and damages for breach of the licensing agreement, 
                claiming loss of market share and competitive advantage. The case involves complex 
                issues of software licensing, intellectual property rights, and contract interpretation 
                in the digital age.
                """,
                "doctrine": "contract_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal"
            },
            {
                "id": "contract_new_002", 
                "title": "Construction Contract Delay - Force Majeure",
                "content": """
                ABC Construction Company contracted with City Municipality to build a new public 
                library with completion deadline of December 31, 2022. Due to supply chain issues 
                caused by the global pandemic and severe weather delays, the project was completed 
                45 days late. The contract included a force majeure clause excusing performance 
                delays due to unforeseeable circumstances beyond the contractor's control. The City 
                seeks to enforce liquidated damages of $1,000 per day for late completion, while 
                ABC Construction argues that the delays were covered by the force majeure provision. 
                The case involves interpretation of force majeure clauses and their application to 
                pandemic-related disruptions.
                """,
                "doctrine": "contract_law",
                "court": "superior_court", 
                "year": 2023,
                "jurisdiction": "state"
            },
            
            # Tort Law Documents
            {
                "id": "tort_new_001",
                "title": "Autonomous Vehicle Accident - Product Liability",
                "content": """
                Plaintiff was injured when an autonomous vehicle manufactured by AutoTech Corp 
                failed to detect a pedestrian crossing and struck the plaintiff at a crosswalk. 
                The vehicle's AI system was operating in full autonomous mode when the accident 
                occurred. Investigation revealed that the vehicle's sensors were functioning 
                properly, but the AI decision-making algorithm failed to properly classify the 
                pedestrian as an obstacle requiring emergency braking. This case presents novel 
                questions about product liability for AI-powered systems, the standard of care 
                for autonomous vehicle manufacturers, and the allocation of responsibility between 
                human operators and artificial intelligence systems.
                """,
                "doctrine": "tort_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal"
            },
            {
                "id": "tort_new_002",
                "title": "Social Media Defamation - Online Reputation",
                "content": """
                Dr. Sarah Mitchell, a prominent physician, filed a defamation lawsuit against 
                former patient John Davis who posted false and damaging reviews about her medical 
                practice on multiple social media platforms and review websites. Davis claimed 
                that Dr. Mitchell provided substandard care and made false statements about her 
                medical credentials. The posts resulted in significant damage to Dr. Mitchell's 
                reputation and loss of patients. The case involves issues of online defamation, 
                the scope of patient review privileges, and the balance between free speech rights 
                and protection from false statements in the digital age.
                """,
                "doctrine": "tort_law",
                "court": "district_court",
                "year": 2023,
                "jurisdiction": "state"
            },
            
            # Constitutional Law Documents
            {
                "id": "constitutional_new_001",
                "title": "Digital Privacy Rights - Fourth Amendment Online",
                "content": """
                Law enforcement obtained defendant's location data from his smartphone without 
                a warrant by requesting the information directly from the cellular service 
                provider. The data revealed defendant's movements over a 30-day period and was 
                used to establish his presence at multiple crime scenes. Defendant argues that 
                warrantless collection of digital location data violates his Fourth Amendment 
                rights and that the digital age requires enhanced privacy protections. The 
                government contends that location data held by third parties is not protected 
                by the Fourth Amendment under the third-party doctrine. This case addresses 
                the intersection of technology and constitutional privacy rights.
                """,
                "doctrine": "constitutional_law",
                "court": "supreme_court",
                "year": 2023,
                "jurisdiction": "federal"
            },
            
            # Criminal Law Documents
            {
                "id": "criminal_new_001",
                "title": "Cybercrime Prosecution - Computer Fraud and Abuse Act",
                "content": """
                Defendant Alex Rodriguez is charged with violating the Computer Fraud and Abuse 
                Act (CFAA) for allegedly accessing his former employer's computer system without 
                authorization and downloading confidential customer data. Rodriguez was terminated 
                from his position as IT administrator, but his access credentials were not 
                immediately revoked. He used these credentials to access the system three days 
                after termination to retrieve what he claims was personal data. The prosecution 
                argues this constitutes unauthorized access under the CFAA, while the defense 
                contends that Rodriguez reasonably believed he had authorization to access his 
                personal files. The case involves interpretation of "authorization" under federal 
                cybercrime statutes.
                """,
                "doctrine": "criminal_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal"
            },
            
            # Employment Law Documents
            {
                "id": "employment_new_001",
                "title": "Remote Work Discrimination - ADA Accommodation",
                "content": """
                Employee Jennifer Park requested to work remotely as a reasonable accommodation 
                for her disability under the Americans with Disabilities Act (ADA). Park suffers 
                from a chronic autoimmune condition that makes her vulnerable to infections and 
                requires frequent medical appointments. Her employer, Global Finance Corp., denied 
                the request, stating that her position requires in-person collaboration and that 
                remote work would impose an undue hardship on business operations. Park filed a 
                complaint alleging disability discrimination and failure to provide reasonable 
                accommodation. The case involves analysis of ADA accommodation requirements in 
                the post-pandemic era where remote work has become more common and technologically 
                feasible.
                """,
                "doctrine": "employment_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal"
            },
            
            # Intellectual Property Documents
            {
                "id": "ip_new_001",
                "title": "Trademark Infringement - Domain Name Dispute",
                "content": """
                Fashion retailer TrendyStyles Inc. filed a trademark infringement lawsuit against 
                online competitor StyleTrend LLC for using a confusingly similar business name 
                and registering the domain name "trendystyles.net". TrendyStyles holds federal 
                trademark registration for "TRENDY STYLES" for clothing retail services and has 
                used the mark in commerce since 2015. StyleTrend argues that their name is 
                sufficiently different and that they operate in a different market segment. The 
                case involves analysis of trademark likelihood of confusion factors, including 
                similarity of marks, relatedness of goods and services, and evidence of actual 
                consumer confusion. The dispute also raises issues under the Anticybersquatting 
                Consumer Protection Act regarding domain name registration.
                """,
                "doctrine": "intellectual_property",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal"
            },
            
            # Corporate Law Documents
            {
                "id": "corporate_new_001",
                "title": "Shareholder Derivative Suit - Executive Compensation",
                "content": """
                Minority shareholders of TechGiant Corp. bring a derivative suit against the board 
                of directors challenging the approval of a $50 million compensation package for 
                the CEO. Plaintiffs allege that the compensation is excessive and not tied to 
                company performance, constituting a breach of the directors' fiduciary duty and 
                waste of corporate assets. The compensation package includes a base salary, 
                performance bonuses, stock options, and a golden parachute provision. Defendants 
                argue that the compensation was approved by an independent compensation committee 
                with advice from external consultants and is necessary to retain top executive 
                talent in a competitive market. The case involves analysis of executive compensation 
                governance and the business judgment rule.
                """,
                "doctrine": "corporate_law",
                "court": "chancery_court",
                "year": 2023,
                "jurisdiction": "state"
            },
            
            # Environmental Law Documents
            {
                "id": "environmental_new_001",
                "title": "Climate Change Litigation - Public Nuisance",
                "content": """
                The City of Coastal Bay filed a lawsuit against major oil companies seeking damages 
                for climate change impacts including sea level rise, increased flooding, and 
                infrastructure damage. The city alleges that defendants knew about the climate 
                risks associated with fossil fuel production but continued to promote fossil fuels 
                while funding climate denial campaigns. The lawsuit is brought under state public 
                nuisance law, seeking compensation for past damages and funding for future climate 
                adaptation measures. Defendants argue that climate change is a global issue that 
                cannot be addressed through tort litigation and that regulation should be left to 
                federal agencies and international agreements. The case represents the growing 
                trend of climate accountability litigation.
                """,
                "doctrine": "environmental_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal"
            }
        ]
        
        print(f"📊 Created {len(additional_documents)} additional legal documents:")
        
        # Count by doctrine
        doctrine_counts = {}
        for doc in additional_documents:
            doctrine = doc['doctrine']
            doctrine_counts[doctrine] = doctrine_counts.get(doctrine, 0) + 1
        
        for doctrine, count in doctrine_counts.items():
            print(f"   • {doctrine.replace('_', ' ').title()}: {count} documents")
        
        return additional_documents
    
    def show_how_to_add_documents(self):
        """Show exactly how to add documents to your system."""
        print("\n🔧 HOW TO ADD THESE DOCUMENTS TO YOUR SYSTEM")
        print("-" * 60)
        
        print("Method 1: Add to working_demo.py")
        print("=" * 30)
        print("1. Open 'working_demo.py' in your editor")
        print("2. Find the 'create_sample_documents()' function")
        print("3. Add new documents to the documents list")
        print("4. Run: python working_demo.py")
        
        print("\nMethod 2: Create your own demo script")
        print("=" * 30)
        print("1. Copy the document structure from above")
        print("2. Create a new Python file")
        print("3. Use the VectorDatabase class to add documents")
        print("4. Test with search queries")
        
        print("\nMethod 3: Use the template below")
        print("=" * 30)
        
        template_code = '''
# Template for adding documents
def add_my_documents():
    from src.vector_db import VectorDatabase
    
    # Initialize vector database
    vector_db = VectorDatabase()
    
    # Your documents
    my_documents = [
        {
            "id": "my_doc_001",
            "title": "My Legal Case Title",
            "content": "Your legal document content here...",
            "doctrine": "contract_law",  # Choose appropriate doctrine
            "court": "district_court",
            "year": 2023,
            "jurisdiction": "federal"
        },
        # Add more documents...
    ]
    
    # Add to database
    success = vector_db.add_documents(my_documents)
    
    if success:
        print(f"✅ Added {len(my_documents)} documents")
        
        # Test search
        results = vector_db.search("your search query", top_k=3)
        for result in results:
            print(f"Found: {result['title']}")
    else:
        print("❌ Failed to add documents")

# Run it
add_my_documents()
        '''
        
        print(template_code)
    
    def demonstrate_search_examples(self):
        """Show search examples with the new documents."""
        print("\n🔍 SEARCH EXAMPLES WITH NEW DOCUMENTS")
        print("-" * 60)
        
        documents = self.create_additional_legal_documents()
        
        # Simulate search results
        search_examples = [
            {
                "query": "software license breach intellectual property",
                "expected_results": ["Software License Agreement Breach", "Trademark Infringement - Domain Name Dispute"]
            },
            {
                "query": "autonomous vehicle accident AI liability",
                "expected_results": ["Autonomous Vehicle Accident - Product Liability"]
            },
            {
                "query": "digital privacy fourth amendment location data",
                "expected_results": ["Digital Privacy Rights - Fourth Amendment Online"]
            },
            {
                "query": "remote work disability accommodation ADA",
                "expected_results": ["Remote Work Discrimination - ADA Accommodation"]
            },
            {
                "query": "climate change environmental litigation damages",
                "expected_results": ["Climate Change Litigation - Public Nuisance"]
            }
        ]
        
        for i, example in enumerate(search_examples, 1):
            print(f"\nExample {i}: '{example['query']}'")
            print("Expected to find:")
            for result in example['expected_results']:
                print(f"   • {result}")
        
        print(f"\n✨ With {len(documents)} additional documents, your search capabilities are greatly enhanced!")
    
    def show_document_statistics(self):
        """Show statistics about the document collection."""
        print("\n📊 DOCUMENT COLLECTION STATISTICS")
        print("-" * 60)
        
        documents = self.create_additional_legal_documents()
        
        # Original documents (5) + Additional documents
        total_original = 5
        total_additional = len(documents)
        total_documents = total_original + total_additional
        
        print(f"📈 Document Collection Growth:")
        print(f"   • Original collection: {total_original} documents")
        print(f"   • Additional documents: {total_additional} documents")
        print(f"   • Total collection: {total_documents} documents")
        print(f"   • Growth: {(total_additional/total_original)*100:.0f}% increase!")
        
        # Legal areas covered
        doctrines = set(doc['doctrine'] for doc in documents)
        print(f"\n📋 Legal Areas Covered:")
        for doctrine in sorted(doctrines):
            print(f"   • {doctrine.replace('_', ' ').title()}")
        
        # Years covered
        years = set(doc['year'] for doc in documents)
        print(f"\n📅 Time Period: {min(years)} - {max(years)}")
        
        # Courts covered
        courts = set(doc['court'] for doc in documents)
        print(f"\n🏛️ Court Levels:")
        for court in sorted(courts):
            print(f"   • {court.replace('_', ' ').title()}")
    
    def run_complete_demo(self):
        """Run complete demonstration of document addition."""
        
        # Create additional documents
        documents = self.create_additional_legal_documents()
        
        # Show how to add them
        self.show_how_to_add_documents()
        
        # Show search examples
        self.demonstrate_search_examples()
        
        # Show statistics
        self.show_document_statistics()
        
        print("\n🎉 DOCUMENT ADDITION DEMO COMPLETE!")
        print("=" * 60)
        print("You now know how to add unlimited legal documents to your system!")
        print("\n📚 Next Steps:")
        print("1. Choose documents relevant to your use case")
        print("2. Add them using one of the methods shown above")
        print("3. Test search functionality with your new documents")
        print("4. Expand your collection as needed")
        
        print(f"\n🚀 Your Vector Database can handle thousands of documents!")
        print(f"Start with these {len(documents)} additional documents and keep growing!")


def main():
    """Main function to run document addition demo."""
    adder = SimpleDocumentAdder()
    adder.run_complete_demo()


if __name__ == "__main__":
    main()
