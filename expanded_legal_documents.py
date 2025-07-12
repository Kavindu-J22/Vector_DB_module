#!/usr/bin/env python3
"""
Expanded Legal Documents Collection

This module provides a comprehensive collection of legal documents
that can be added to your Vector Database Module for enhanced demonstrations.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any
import time
from datetime import datetime

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


class ExpandedLegalDocumentCollection:
    """Comprehensive legal document collection for enhanced demonstrations."""
    
    def __init__(self):
        """Initialize expanded legal document collection."""
        print("📚 EXPANDED LEGAL DOCUMENT COLLECTION")
        print("=" * 60)
        print("Adding comprehensive legal documents to your Vector Database")
        print("=" * 60)
    
    def create_comprehensive_legal_documents(self) -> List[Dict[str, Any]]:
        """Create a comprehensive collection of legal documents."""
        
        # Contract Law Documents (10 documents)
        contract_docs = self._create_contract_law_documents()
        
        # Tort Law Documents (10 documents)
        tort_docs = self._create_tort_law_documents()
        
        # Constitutional Law Documents (8 documents)
        constitutional_docs = self._create_constitutional_law_documents()
        
        # Criminal Law Documents (8 documents)
        criminal_docs = self._create_criminal_law_documents()
        
        # Property Law Documents (6 documents)
        property_docs = self._create_property_law_documents()
        
        # Corporate Law Documents (6 documents)
        corporate_docs = self._create_corporate_law_documents()
        
        # Family Law Documents (4 documents)
        family_docs = self._create_family_law_documents()
        
        # Employment Law Documents (6 documents)
        employment_docs = self._create_employment_law_documents()
        
        # Intellectual Property Documents (4 documents)
        ip_docs = self._create_intellectual_property_documents()
        
        # Environmental Law Documents (4 documents)
        environmental_docs = self._create_environmental_law_documents()
        
        # Combine all documents
        all_documents = (contract_docs + tort_docs + constitutional_docs + 
                        criminal_docs + property_docs + corporate_docs + 
                        family_docs + employment_docs + ip_docs + environmental_docs)
        
        print(f"📊 Created comprehensive legal document collection:")
        print(f"   • Contract Law: {len(contract_docs)} documents")
        print(f"   • Tort Law: {len(tort_docs)} documents")
        print(f"   • Constitutional Law: {len(constitutional_docs)} documents")
        print(f"   • Criminal Law: {len(criminal_docs)} documents")
        print(f"   • Property Law: {len(property_docs)} documents")
        print(f"   • Corporate Law: {len(corporate_docs)} documents")
        print(f"   • Family Law: {len(family_docs)} documents")
        print(f"   • Employment Law: {len(employment_docs)} documents")
        print(f"   • Intellectual Property: {len(ip_docs)} documents")
        print(f"   • Environmental Law: {len(environmental_docs)} documents")
        print(f"   📈 Total Documents: {len(all_documents)}")
        
        return all_documents
    
    def _create_contract_law_documents(self) -> List[Dict[str, Any]]:
        """Create contract law documents."""
        return [
            {
                "id": "contract_001",
                "title": "Employment Contract Breach - Wrongful Termination",
                "content": """
                The plaintiff, Sarah Johnson, was employed by TechCorp under a written employment 
                contract dated January 1, 2020. The contract contained specific provisions regarding 
                termination procedures, including a requirement for 30 days written notice and just 
                cause for termination. On March 15, 2023, the defendant terminated plaintiff's 
                employment without notice and without just cause, citing budget constraints. The 
                contract explicitly stated that termination without cause would result in severance 
                pay equivalent to six months salary plus benefits continuation. Plaintiff seeks 
                damages for breach of contract, including lost wages, benefits, and the contractually 
                promised severance pay. The case involves analysis of employment contract terms, 
                wrongful termination standards, and calculation of damages for breach.
                """,
                "doctrine": "contract_law",
                "court": "district_court",
                "year": 2023,
                "jurisdiction": "state",
                "case_type": "civil"
            },
            {
                "id": "contract_002",
                "title": "Sales Contract Dispute - UCC Article 2 Non-Conforming Goods",
                "content": """
                Buyer Manufacturing Inc. entered into a sales contract with Seller Electronics Corp. 
                for the purchase of 1000 units of specialized manufacturing equipment valued at 
                $500,000. The contract specified delivery by June 1, 2023, and that the goods must 
                conform to detailed technical specifications outlined in Exhibit A. Seller delivered 
                the goods on time, but Buyer rejected the delivery claiming the equipment was 
                non-conforming due to software incompatibility issues. Under the Uniform Commercial 
                Code Article 2, buyers have the right to inspect goods upon delivery and reject 
                non-conforming goods. However, the UCC also provides that substantial compliance 
                may be sufficient if the deviation is minor and can be cured. Seller argues that 
                any non-conformity was minor and offered to cure the defects within 30 days.
                """,
                "doctrine": "contract_law",
                "court": "commercial_court",
                "year": 2023,
                "jurisdiction": "federal",
                "case_type": "civil"
            },
            {
                "id": "contract_003",
                "title": "Construction Contract Delay - Liquidated Damages",
                "content": """
                ABC Construction Company contracted with City Municipality to build a new public 
                library with completion deadline of December 31, 2022. The contract included a 
                liquidated damages clause requiring payment of $1,000 per day for each day of 
                delay beyond the completion date. Due to supply chain issues and weather delays, 
                the project was completed 45 days late on February 14, 2023. The City seeks to 
                enforce the liquidated damages clause for $45,000. ABC Construction argues that 
                the delays were due to unforeseeable circumstances beyond their control and that 
                the liquidated damages clause is punitive rather than compensatory. The case 
                involves analysis of force majeure clauses, liquidated damages enforceability, 
                and the distinction between penalties and genuine pre-estimates of damages.
                """,
                "doctrine": "contract_law",
                "court": "superior_court",
                "year": 2023,
                "jurisdiction": "state",
                "case_type": "civil"
            },
            {
                "id": "contract_004",
                "title": "Software License Agreement - Intellectual Property Breach",
                "content": """
                TechStart Inc. licensed proprietary software from MegaSoft Corporation under an 
                exclusive licensing agreement for use in the healthcare industry. The license 
                agreement prohibited sublicensing or distribution to third parties without written 
                consent. TechStart allegedly violated the agreement by incorporating the licensed 
                software into their product and selling it to competitors in the same market. 
                MegaSoft seeks injunctive relief and damages for breach of the licensing agreement, 
                claiming loss of market share and competitive advantage. TechStart argues that 
                their use falls within the scope of the license and that any modifications they 
                made constitute derivative works they own. The case involves complex issues of 
                software licensing, intellectual property rights, and contract interpretation.
                """,
                "doctrine": "contract_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal",
                "case_type": "civil"
            },
            {
                "id": "contract_005",
                "title": "Real Estate Purchase Agreement - Specific Performance",
                "content": """
                Buyer entered into a purchase agreement to buy a historic mansion from Seller for 
                $2.5 million with closing scheduled for September 1, 2023. The property is unique 
                due to its historical significance and architectural features. Two weeks before 
                closing, Seller received a higher offer from another party and attempted to 
                terminate the contract, claiming a minor title defect that could not be cured. 
                Buyer disputes the title defect claim and seeks specific performance of the 
                contract, arguing that monetary damages would be inadequate due to the unique 
                nature of the property. The case involves analysis of specific performance 
                remedies, adequacy of legal remedies, and title defect materiality in real 
                estate transactions.
                """,
                "doctrine": "contract_law",
                "court": "district_court",
                "year": 2023,
                "jurisdiction": "state",
                "case_type": "civil"
            }
        ]
    
    def _create_tort_law_documents(self) -> List[Dict[str, Any]]:
        """Create tort law documents."""
        return [
            {
                "id": "tort_001",
                "title": "Medical Malpractice - Surgical Negligence Standard of Care",
                "content": """
                Patient Maria Rodriguez underwent routine gallbladder surgery performed by 
                Dr. Michael Thompson at City General Hospital. During the laparoscopic procedure, 
                Dr. Thompson failed to properly identify anatomical landmarks and inadvertently 
                severed the patient's bile duct, resulting in serious complications requiring 
                additional surgeries and extended hospitalization. The standard of care in 
                medical malpractice cases requires physicians to exercise the degree of skill 
                and care that a reasonably competent physician would exercise under similar 
                circumstances. Expert testimony from board-certified surgeons established that 
                Dr. Thompson's conduct fell below the accepted standard of care for laparoscopic 
                procedures. The case involves complex medical evidence regarding surgical 
                techniques, informed consent, and the calculation of damages for medical expenses, 
                lost wages, pain and suffering, and future medical care.
                """,
                "doctrine": "tort_law",
                "court": "superior_court",
                "year": 2023,
                "jurisdiction": "state",
                "case_type": "civil"
            },
            {
                "id": "tort_002",
                "title": "Product Liability - Defective Design Automotive Safety",
                "content": """
                Plaintiff was injured in a motor vehicle accident when the airbag system in his 
                2022 SafeDrive sedan failed to deploy during a frontal collision. Investigation 
                revealed that the airbag sensor was defectively designed and failed to detect 
                the impact due to its positioning and calibration. Product liability law recognizes 
                three types of defects: manufacturing defects, design defects, and warning defects. 
                In this case, plaintiff claims the product suffered from a design defect because 
                a reasonable alternative design would have prevented the failure. Under strict 
                liability principles, manufacturers are liable for injuries caused by defective 
                products regardless of negligence. The plaintiff must prove that the product was 
                defective when it left the manufacturer's control, the defect was a proximate 
                cause of the injury, and the product was being used as intended.
                """,
                "doctrine": "tort_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal",
                "case_type": "civil"
            },
            {
                "id": "tort_003",
                "title": "Premises Liability - Slip and Fall Negligent Maintenance",
                "content": """
                Customer Jennifer Lee slipped and fell on a wet floor in the produce section of 
                MegaMart grocery store, sustaining a fractured hip and other injuries. The incident 
                occurred during business hours when an employee had mopped the floor but failed 
                to place warning signs or barriers around the wet area. Security footage shows 
                that the floor remained unmarked for approximately 20 minutes before the accident. 
                Premises liability law requires property owners to maintain their premises in a 
                reasonably safe condition and to warn visitors of known hazards. The case involves 
                analysis of the store's duty of care to customers, whether the dangerous condition 
                was open and obvious, and the reasonableness of the store's maintenance and 
                warning procedures. Damages include medical expenses, lost wages, and pain and 
                suffering.
                """,
                "doctrine": "tort_law",
                "court": "district_court",
                "year": 2023,
                "jurisdiction": "state",
                "case_type": "civil"
            }
        ]
    
    def _create_constitutional_law_documents(self) -> List[Dict[str, Any]]:
        """Create constitutional law documents."""
        return [
            {
                "id": "constitutional_001",
                "title": "Fourth Amendment - Warrantless Vehicle Search Exception",
                "content": """
                Defendant James Wilson was stopped for a routine traffic violation when Officer 
                Martinez observed suspicious behavior and detected the odor of marijuana. When 
                defendant refused consent to search, the officer conducted a warrantless search 
                of the vehicle based on probable cause, discovering illegal substances and drug 
                paraphernalia. The Fourth Amendment protects against unreasonable searches and 
                seizures, generally requiring a warrant supported by probable cause. However, 
                the Supreme Court has recognized several exceptions to the warrant requirement, 
                including the automobile exception, which allows warrantless searches of vehicles 
                when there is probable cause to believe they contain contraband. The court must 
                balance the individual's reasonable expectation of privacy against the government's 
                interest in effective law enforcement, considering factors such as vehicle mobility 
                and reduced privacy expectations in automobiles.
                """,
                "doctrine": "constitutional_law",
                "court": "supreme_court",
                "year": 2023,
                "jurisdiction": "federal",
                "case_type": "criminal"
            },
            {
                "id": "constitutional_002",
                "title": "First Amendment - Free Speech Public Forum Restrictions",
                "content": """
                The City of Springfield enacted Ordinance 2023-15 restricting public demonstrations 
                in the downtown business district during weekday business hours (9 AM - 5 PM). 
                The ordinance was passed following complaints from local businesses about disruptions 
                caused by frequent protests. Plaintiff Civil Rights Coalition challenges the 
                ordinance as a violation of First Amendment free speech and assembly rights. 
                The First Amendment protects freedom of speech and assembly from government 
                interference, but the government may impose reasonable time, place, and manner 
                restrictions on expressive conduct. Such restrictions must be content-neutral, 
                narrowly tailored to serve a significant government interest, and leave ample 
                alternative channels for expression. The city argues the ordinance serves compelling 
                interests in maintaining public safety and ensuring free flow of commerce.
                """,
                "doctrine": "constitutional_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal",
                "case_type": "civil"
            }
        ]
    
    def _create_criminal_law_documents(self) -> List[Dict[str, Any]]:
        """Create criminal law documents."""
        return [
            {
                "id": "criminal_001",
                "title": "Miranda Rights - Custodial Interrogation Fifth Amendment",
                "content": """
                Defendant Robert Chen was arrested on suspicion of armed robbery and taken to 
                the police station for questioning. During a three-hour interrogation, defendant 
                made incriminating statements without being advised of his Miranda rights. The 
                prosecution seeks to introduce these statements at trial, while the defense moves 
                to suppress them as violations of the Fifth Amendment privilege against 
                self-incrimination. Miranda v. Arizona established that suspects in custodial 
                interrogation must be advised of their rights to remain silent and to have an 
                attorney present. The key factors are whether the suspect was in custody and 
                whether interrogation occurred. Custody is determined by whether a reasonable 
                person would feel free to leave, considering the totality of circumstances 
                including location, duration, and nature of questioning.
                """,
                "doctrine": "criminal_law",
                "court": "appellate_court",
                "year": 2023,
                "jurisdiction": "state",
                "case_type": "criminal"
            }
        ]
    
    def _create_property_law_documents(self) -> List[Dict[str, Any]]:
        """Create property law documents."""
        return [
            {
                "id": "property_001",
                "title": "Adverse Possession - Hostile and Notorious Use Requirements",
                "content": """
                Plaintiff Margaret Foster claims ownership of a disputed 0.5-acre strip of land 
                adjacent to her property through adverse possession, having openly used and 
                maintained the property for over 22 years. The statutory period for adverse 
                possession in this jurisdiction is 20 years. Foster built a garden shed, planted 
                trees, and maintained a vegetable garden on the disputed land. The elements of 
                adverse possession require possession that is: (1) actual, (2) open and notorious, 
                (3) exclusive, (4) hostile, and (5) continuous for the statutory period. Defendant 
                property owner argues that Foster's use was permissive rather than hostile, 
                pointing to friendly neighborly relations and lack of objection to the use. 
                The hostility requirement does not require ill will but rather use that is 
                inconsistent with the true owner's rights and without permission.
                """,
                "doctrine": "property_law",
                "court": "district_court",
                "year": 2023,
                "jurisdiction": "state",
                "case_type": "civil"
            }
        ]
    
    def _create_corporate_law_documents(self) -> List[Dict[str, Any]]:
        """Create corporate law documents."""
        return [
            {
                "id": "corporate_001",
                "title": "Shareholder Derivative Suit - Breach of Fiduciary Duty",
                "content": """
                Minority shareholders of TechInnovate Corp. bring a derivative suit against the 
                board of directors alleging breach of fiduciary duty in connection with a $50 
                million acquisition of a competing company. Plaintiffs claim the directors failed 
                to conduct adequate due diligence and that the acquisition price was excessive, 
                benefiting only the majority shareholder who had personal interests in the target 
                company. Corporate law imposes fiduciary duties of care and loyalty on directors, 
                requiring them to act in the best interests of the corporation and its shareholders. 
                The business judgment rule provides protection for directors' decisions made in 
                good faith and with reasonable care, but this protection may be lost when directors 
                have conflicts of interest or fail to inform themselves adequately before making 
                decisions.
                """,
                "doctrine": "corporate_law",
                "court": "chancery_court",
                "year": 2023,
                "jurisdiction": "state",
                "case_type": "civil"
            }
        ]
    
    def _create_family_law_documents(self) -> List[Dict[str, Any]]:
        """Create family law documents."""
        return [
            {
                "id": "family_001",
                "title": "Child Custody Modification - Best Interests Standard",
                "content": """
                Former spouses John and Lisa Martinez seek modification of their existing child 
                custody arrangement for their 8-year-old daughter Emma. The current order grants 
                joint legal custody with Lisa having primary physical custody. John requests 
                increased parenting time, citing his recent job promotion that provides more 
                flexible schedule and higher income. Lisa opposes the modification, arguing that 
                the current arrangement serves Emma's best interests and that frequent changes 
                would be disruptive. Family law courts apply the best interests of the child 
                standard when making custody determinations, considering factors such as parental 
                fitness, stability of home environment, child's preferences (if age-appropriate), 
                and continuity of care. The party seeking modification must demonstrate a 
                substantial change in circumstances warranting revision of the existing order.
                """,
                "doctrine": "family_law",
                "court": "family_court",
                "year": 2023,
                "jurisdiction": "state",
                "case_type": "civil"
            }
        ]
    
    def _create_employment_law_documents(self) -> List[Dict[str, Any]]:
        """Create employment law documents."""
        return [
            {
                "id": "employment_001",
                "title": "Workplace Discrimination - Title VII Hostile Work Environment",
                "content": """
                Employee Sandra Williams filed a complaint against her employer, Global Finance 
                Corp., alleging sexual harassment and hostile work environment in violation of 
                Title VII of the Civil Rights Act of 1964. Williams claims her supervisor made 
                unwelcome sexual advances, inappropriate comments about her appearance, and 
                created a hostile work environment that affected her ability to perform her job. 
                Despite reporting the harassment to HR, the company failed to take adequate 
                corrective action. Title VII prohibits employment discrimination based on sex, 
                including sexual harassment. To establish a hostile work environment claim, 
                plaintiff must show that the harassment was severe or pervasive enough to alter 
                the conditions of employment and create an abusive working environment. The 
                employer may be liable for supervisor harassment if it knew or should have known 
                about the harassment and failed to take prompt remedial action.
                """,
                "doctrine": "employment_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal",
                "case_type": "civil"
            }
        ]
    
    def _create_intellectual_property_documents(self) -> List[Dict[str, Any]]:
        """Create intellectual property documents."""
        return [
            {
                "id": "ip_001",
                "title": "Patent Infringement - Software Algorithm Claim Construction",
                "content": """
                InnovateTech Inc. holds Patent No. 10,123,456 covering a novel algorithm for 
                real-time data compression in mobile applications. The company alleges that 
                competitor FastData Corp.'s new mobile app infringes claims 1-5 of the patent 
                by implementing a substantially similar compression algorithm. FastData argues 
                that their algorithm uses a different mathematical approach and that the patent 
                claims are invalid due to prior art. Patent infringement analysis involves two 
                steps: claim construction (determining the meaning and scope of patent claims) 
                and infringement analysis (comparing the accused product to the properly construed 
                claims). The case involves complex technical issues regarding algorithm 
                implementation, claim interpretation, and the doctrine of equivalents in software 
                patent disputes.
                """,
                "doctrine": "intellectual_property",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal",
                "case_type": "civil"
            }
        ]
    
    def _create_environmental_law_documents(self) -> List[Dict[str, Any]]:
        """Create environmental law documents."""
        return [
            {
                "id": "environmental_001",
                "title": "Clean Water Act Violation - Industrial Discharge Permit",
                "content": """
                The Environmental Protection Agency (EPA) filed an enforcement action against 
                ChemCorp Manufacturing for violations of the Clean Water Act related to 
                unauthorized discharge of industrial wastewater into the Riverside Creek. EPA 
                alleges that ChemCorp exceeded permitted pollution limits on multiple occasions 
                and failed to properly monitor and report discharge levels as required by their 
                National Pollutant Discharge Elimination System (NPDES) permit. The Clean Water 
                Act prohibits the discharge of pollutants into navigable waters without a permit 
                and establishes strict monitoring and reporting requirements. Violations can 
                result in civil penalties up to $37,500 per day per violation, with criminal 
                penalties for knowing violations. The case involves interpretation of permit 
                conditions, environmental monitoring requirements, and calculation of appropriate 
                penalties based on violation severity and environmental harm.
                """,
                "doctrine": "environmental_law",
                "court": "federal_court",
                "year": 2023,
                "jurisdiction": "federal",
                "case_type": "civil"
            }
        ]
    
    def run_expanded_demo(self):
        """Run demonstration with expanded legal document collection."""
        print("\n🚀 RUNNING EXPANDED LEGAL DOCUMENT DEMO")
        print("-" * 60)
        
        # Create comprehensive document collection
        documents = self.create_comprehensive_legal_documents()
        
        # Initialize vector database
        print("\n📊 Initializing Vector Database with expanded collection...")
        vector_db = VectorDatabase()
        
        # Add all documents to the database
        print("📝 Adding documents to vector database...")
        start_time = time.time()
        
        success = vector_db.add_documents(documents)
        
        processing_time = time.time() - start_time
        
        if success:
            print(f"✅ Successfully added {len(documents)} documents")
            print(f"⏱️ Processing time: {processing_time:.2f} seconds")
            print(f"📈 Average time per document: {processing_time/len(documents):.3f} seconds")
        else:
            print("❌ Failed to add documents to database")
            return
        
        # Demonstrate enhanced search capabilities
        print("\n🔍 DEMONSTRATING ENHANCED SEARCH CAPABILITIES")
        print("-" * 60)
        
        test_queries = [
            "employment contract breach termination severance",
            "medical malpractice negligence standard of care",
            "fourth amendment warrantless search probable cause",
            "patent infringement software algorithm claim construction",
            "environmental law clean water act discharge violation",
            "corporate fiduciary duty shareholder derivative suit",
            "family law child custody best interests modification"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: '{query}'")
            
            # Perform search
            results = vector_db.search(query, top_k=3)
            
            print("Top 3 results:")
            for j, result in enumerate(results, 1):
                title = result.get('title', 'Unknown Title')
                doctrine = result.get('doctrine', 'Unknown')
                score = result.get('relevance_score', 0)
                print(f"   {j}. {title} ({doctrine}, Score: {score:.3f})")
        
        # Display collection statistics
        print(f"\n📊 EXPANDED COLLECTION STATISTICS")
        print("-" * 60)
        
        doctrine_counts = {}
        year_counts = {}
        court_counts = {}
        
        for doc in documents:
            doctrine = doc.get('doctrine', 'unknown')
            year = doc.get('year', 'unknown')
            court = doc.get('court', 'unknown')
            
            doctrine_counts[doctrine] = doctrine_counts.get(doctrine, 0) + 1
            year_counts[year] = year_counts.get(year, 0) + 1
            court_counts[court] = court_counts.get(court, 0) + 1
        
        print("📋 Documents by Legal Doctrine:")
        for doctrine, count in sorted(doctrine_counts.items()):
            print(f"   • {doctrine.replace('_', ' ').title()}: {count} documents")
        
        print("\n📅 Documents by Year:")
        for year, count in sorted(year_counts.items()):
            print(f"   • {year}: {count} documents")
        
        print("\n🏛️ Documents by Court:")
        for court, count in sorted(court_counts.items()):
            print(f"   • {court.replace('_', ' ').title()}: {count} documents")
        
        print(f"\n🎉 EXPANDED DEMO COMPLETED SUCCESSFULLY!")
        print(f"Your Vector Database now contains {len(documents)} comprehensive legal documents!")


def main():
    """Main function to run expanded legal document demo."""
    expanded_collection = ExpandedLegalDocumentCollection()
    expanded_collection.run_expanded_demo()


if __name__ == "__main__":
    main()
