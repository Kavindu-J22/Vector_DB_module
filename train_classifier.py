#!/usr/bin/env python3
"""
Training Script for Legal Document Classifier

This script trains BERT-based classifiers for legal document categorization.
It supports training separate models for doctrine and court level classification.

Usage:
    python train_classifier.py --data-path data/labeled_documents.json
    python train_classifier.py --data-path data/labeled_documents.json --model-type doctrine
    python train_classifier.py --data-path data/labeled_documents.json --model-type court
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.classifier import DocumentClassifier
from src.utils import load_config, setup_logging, create_directories


def load_training_data(data_path: str):
    """Load training data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training documents from {data_path}")
    return data


def validate_training_data(documents):
    """Validate training data format."""
    required_fields = ['text', 'labels']
    
    for i, doc in enumerate(documents):
        if not all(field in doc for field in required_fields):
            raise ValueError(f"Document {i} missing required fields: {required_fields}")
        
        if not isinstance(doc['labels'], dict):
            raise ValueError(f"Document {i} labels must be a dictionary")
    
    print("Training data validation passed")


def analyze_training_data(documents):
    """Analyze training data distribution."""
    doctrine_counts = {}
    court_counts = {}
    year_counts = {}
    
    for doc in documents:
        labels = doc['labels']
        
        # Count doctrines
        if 'doctrine' in labels:
            doctrine = labels['doctrine']
            doctrine_counts[doctrine] = doctrine_counts.get(doctrine, 0) + 1
        
        # Count courts
        if 'court' in labels:
            court = labels['court']
            court_counts[court] = court_counts.get(court, 0) + 1
        
        # Count years
        if 'year' in labels:
            year = labels['year']
            year_counts[year] = year_counts.get(year, 0) + 1
    
    print("\nTraining Data Analysis:")
    print("-" * 30)
    
    if doctrine_counts:
        print("Doctrine distribution:")
        for doctrine, count in sorted(doctrine_counts.items()):
            print(f"  {doctrine}: {count}")
    
    if court_counts:
        print("\nCourt distribution:")
        for court, count in sorted(court_counts.items()):
            print(f"  {court}: {count}")
    
    if year_counts:
        print(f"\nYear range: {min(year_counts.keys())} - {max(year_counts.keys())}")
        print(f"Total documents with years: {sum(year_counts.values())}")
    
    return doctrine_counts, court_counts, year_counts


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Legal Document Classifier")
    parser.add_argument("--data-path", required=True, 
                       help="Path to training data JSON file")
    parser.add_argument("--model-type", choices=["doctrine", "court", "both"], 
                       default="both", help="Type of classifier to train")
    parser.add_argument("--config-path", default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", default="data/models",
                       help="Directory to save trained models")
    
    args = parser.parse_args()
    
    print("="*60)
    print("LEGAL DOCUMENT CLASSIFIER TRAINING")
    print("="*60)
    
    # Setup
    config = load_config(args.config_path)
    setup_logging(config)
    
    # Create output directory
    create_directories([args.output_dir])
    
    # Load and validate training data
    print("\n1. LOADING TRAINING DATA")
    print("-" * 30)
    
    documents = load_training_data(args.data_path)
    validate_training_data(documents)
    
    # Analyze data distribution
    doctrine_counts, court_counts, year_counts = analyze_training_data(documents)
    
    # Initialize classifier
    print("\n2. INITIALIZING CLASSIFIER")
    print("-" * 32)
    
    classifier = DocumentClassifier(args.config_path)
    
    # Prepare training data
    print("\n3. PREPARING TRAINING DATA")
    print("-" * 33)
    
    doctrine_data, court_data = classifier.prepare_training_data(documents)
    
    if not doctrine_data and not court_data:
        print("Error: No valid training data found!")
        print("Make sure your documents have 'doctrine' and/or 'court' labels")
        return
    
    # Train models based on selection
    if args.model_type in ["doctrine", "both"] and doctrine_data:
        print("\n4. TRAINING DOCTRINE CLASSIFIER")
        print("-" * 36)
        
        print(f"Training samples: {len(doctrine_data['train_texts'])}")
        print(f"Validation samples: {len(doctrine_data['val_texts'])}")
        print(f"Number of doctrine classes: {len(set(doctrine_data['train_labels']))}")
        
        classifier.train_classifier(doctrine_data, model_type="doctrine")
        print("Doctrine classifier training completed!")
    
    if args.model_type in ["court", "both"] and court_data:
        print("\n5. TRAINING COURT CLASSIFIER")
        print("-" * 32)
        
        print(f"Training samples: {len(court_data['train_texts'])}")
        print(f"Validation samples: {len(court_data['val_texts'])}")
        print(f"Number of court classes: {len(set(court_data['train_labels']))}")
        
        classifier.train_classifier(court_data, model_type="court")
        print("Court classifier training completed!")
    
    # Test the trained models
    print("\n6. TESTING TRAINED MODELS")
    print("-" * 30)
    
    # Load the trained models
    try:
        classifier.load_trained_models()
        
        # Test with a sample document
        test_doc = documents[0]
        test_text = test_doc['text']
        
        print(f"Testing with sample document...")
        print(f"Text preview: {test_text[:200]}...")
        
        result = classifier.classify_document(test_text)
        
        print(f"\nClassification Results:")
        print(f"Predicted doctrine: {result.doctrine} (confidence: {result.confidence_scores['doctrine']:.3f})")
        print(f"Predicted court: {result.court_level} (confidence: {result.confidence_scores['court']:.3f})")
        print(f"Extracted year: {result.year}")
        
        # Compare with ground truth if available
        if 'labels' in test_doc:
            true_labels = test_doc['labels']
            print(f"\nGround Truth:")
            if 'doctrine' in true_labels:
                print(f"True doctrine: {true_labels['doctrine']}")
            if 'court' in true_labels:
                print(f"True court: {true_labels['court']}")
            if 'year' in true_labels:
                print(f"True year: {true_labels['year']}")
        
    except Exception as e:
        print(f"Error testing trained models: {e}")
        print("Models may not have been saved properly")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"\nTrained models saved to:")
    if args.model_type in ["doctrine", "both"] and doctrine_data:
        print(f"- Doctrine classifier: {args.output_dir}/doctrine_classifier_final/")
    if args.model_type in ["court", "both"] and court_data:
        print(f"- Court classifier: {args.output_dir}/court_classifier_final/")
    
    print(f"\nEvaluation results saved to:")
    if args.model_type in ["doctrine", "both"] and doctrine_data:
        print(f"- Doctrine evaluation: {args.output_dir}/doctrine_eval_results.json")
    if args.model_type in ["court", "both"] and court_data:
        print(f"- Court evaluation: {args.output_dir}/court_eval_results.json")
    
    print(f"\nTraining logs available in: logs/")
    
    print("\nNext steps:")
    print("1. Review evaluation results to assess model performance")
    print("2. Use the trained models in your retrieval system")
    print("3. Fine-tune hyperparameters if needed")
    print("4. Collect more training data for underrepresented classes")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nTraining script finished.")
