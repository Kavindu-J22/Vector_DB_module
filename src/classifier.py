"""
Document Classification System for Legal Documents

This module implements a BERT-based classifier to categorize legal documents
by doctrine, court level, and year for metadata tagging.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from torch.utils.data import Dataset
from loguru import logger
import re
from datetime import datetime

from .utils import load_config, save_json, load_json, save_pickle, load_pickle


@dataclass
class ClassificationResult:
    """Result of document classification."""
    doctrine: str
    court_level: str
    year: Optional[int]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]


class LegalDocumentDataset(Dataset):
    """Dataset class for legal document classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DocumentClassifier:
    """
    BERT-based classifier for legal document categorization.
    
    Classifies documents by:
    - Doctrine (contract, tort, property, etc.)
    - Court level (supreme, appeal, district, etc.)
    - Year (extracted from text)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the classifier with configuration."""
        self.config = load_config(config_path)
        self.classifier_config = self.config['models']['classifier']
        self.data_config = self.config['data']
        
        # Initialize tokenizer and models
        self.tokenizer = None
        self.doctrine_model = None
        self.court_model = None
        
        # Label mappings
        self.doctrine_labels = self.data_config['doctrines']
        self.court_labels = self.data_config['court_levels']
        self.doctrine_label_to_id = {label: idx for idx, label in enumerate(self.doctrine_labels)}
        self.court_label_to_id = {label: idx for idx, label in enumerate(self.court_labels)}
        self.doctrine_id_to_label = {idx: label for label, idx in self.doctrine_label_to_id.items()}
        self.court_id_to_label = {idx: label for label, idx in self.court_label_to_id.items()}
        
        logger.info("DocumentClassifier initialized")
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract year from legal document text."""
        # Common patterns for years in legal documents
        year_patterns = [
            r'\b(19|20)\d{2}\b',  # 4-digit years
            r'\b\d{1,2}/\d{1,2}/(19|20)\d{2}\b',  # Date formats
            r'\b(19|20)\d{2}\s*(?:decision|ruling|case|judgment)\b',  # Year with legal terms
        ]
        
        years = []
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Extract year from tuple (from grouped patterns)
                    year_str = ''.join(match)
                    if len(year_str) == 4:
                        years.append(int(year_str))
                else:
                    # Extract 4-digit year
                    year_match = re.search(r'\b(19|20)\d{2}\b', match)
                    if year_match:
                        years.append(int(year_match.group()))
        
        if years:
            # Return the most recent reasonable year
            valid_years = [y for y in years if 1900 <= y <= datetime.now().year]
            return max(valid_years) if valid_years else None
        
        return None
    
    def _initialize_models(self):
        """Initialize tokenizer and models."""
        model_name = self.classifier_config['model_name']
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize models for different classification tasks
        self.doctrine_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.doctrine_labels)
        )
        
        self.court_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.court_labels)
        )
        
        logger.info(f"Models initialized with {model_name}")
    
    def prepare_training_data(self, documents: List[Dict[str, Any]]) -> Tuple[Dict, Dict]:
        """
        Prepare training data from annotated documents.
        
        Args:
            documents: List of documents with 'text' and 'labels' fields
            
        Returns:
            Tuple of (doctrine_data, court_data) dictionaries
        """
        doctrine_texts, doctrine_labels = [], []
        court_texts, court_labels = [], []
        
        for doc in documents:
            text = doc['text']
            labels = doc['labels']
            
            # Prepare doctrine classification data
            if 'doctrine' in labels:
                doctrine = labels['doctrine']
                if doctrine in self.doctrine_label_to_id:
                    doctrine_texts.append(text)
                    doctrine_labels.append(self.doctrine_label_to_id[doctrine])
            
            # Prepare court level classification data
            if 'court' in labels:
                court = labels['court']
                if court in self.court_label_to_id:
                    court_texts.append(text)
                    court_labels.append(self.court_label_to_id[court])
        
        # Split data
        doctrine_data = {}
        if doctrine_texts:
            doctrine_train_texts, doctrine_val_texts, doctrine_train_labels, doctrine_val_labels = train_test_split(
                doctrine_texts, doctrine_labels, test_size=0.2, random_state=42, stratify=doctrine_labels
            )
            doctrine_data = {
                'train_texts': doctrine_train_texts,
                'val_texts': doctrine_val_texts,
                'train_labels': doctrine_train_labels,
                'val_labels': doctrine_val_labels
            }
        
        court_data = {}
        if court_texts:
            court_train_texts, court_val_texts, court_train_labels, court_val_labels = train_test_split(
                court_texts, court_labels, test_size=0.2, random_state=42, stratify=court_labels
            )
            court_data = {
                'train_texts': court_train_texts,
                'val_texts': court_val_texts,
                'train_labels': court_train_labels,
                'val_labels': court_val_labels
            }
        
        logger.info(f"Prepared training data: {len(doctrine_texts)} doctrine samples, {len(court_texts)} court samples")
        return doctrine_data, court_data
    
    def train_classifier(self, training_data: Dict, model_type: str = "doctrine") -> None:
        """
        Train a classifier model.
        
        Args:
            training_data: Dictionary with train/val texts and labels
            model_type: Either "doctrine" or "court"
        """
        if not training_data:
            logger.warning(f"No training data provided for {model_type} classifier")
            return
        
        if self.tokenizer is None:
            self._initialize_models()
        
        # Select the appropriate model
        model = self.doctrine_model if model_type == "doctrine" else self.court_model
        
        # Create datasets
        train_dataset = LegalDocumentDataset(
            training_data['train_texts'],
            training_data['train_labels'],
            self.tokenizer,
            self.classifier_config['max_length']
        )
        
        val_dataset = LegalDocumentDataset(
            training_data['val_texts'],
            training_data['val_labels'],
            self.tokenizer,
            self.classifier_config['max_length']
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./data/models/{model_type}_classifier',
            num_train_epochs=self.classifier_config['num_epochs'],
            per_device_train_batch_size=self.classifier_config['batch_size'],
            per_device_eval_batch_size=self.classifier_config['batch_size'],
            learning_rate=self.classifier_config['learning_rate'],
            warmup_steps=100,
            logging_dir=f'./logs/{model_type}_training',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        logger.info(f"Starting training for {model_type} classifier")
        trainer.train()
        
        # Save the model
        model_path = f'./data/models/{model_type}_classifier_final'
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Evaluate the model
        eval_results = trainer.evaluate()
        logger.info(f"{model_type} classifier evaluation results: {eval_results}")
        
        # Save evaluation results
        save_json(eval_results, f'./data/models/{model_type}_eval_results.json')
        
        logger.info(f"{model_type} classifier training completed")
    
    def load_trained_models(self, doctrine_model_path: str = None, court_model_path: str = None):
        """Load pre-trained classifier models."""
        if doctrine_model_path is None:
            doctrine_model_path = './data/models/doctrine_classifier_final'
        if court_model_path is None:
            court_model_path = './data/models/court_classifier_final'
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(doctrine_model_path)
            
            # Load models
            self.doctrine_model = AutoModelForSequenceClassification.from_pretrained(doctrine_model_path)
            self.court_model = AutoModelForSequenceClassification.from_pretrained(court_model_path)
            
            # Set models to evaluation mode
            self.doctrine_model.eval()
            self.court_model.eval()
            
            logger.info("Pre-trained models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading pre-trained models: {e}")
            logger.info("Initializing new models instead")
            self._initialize_models()
    
    def classify_document(self, text: str) -> ClassificationResult:
        """
        Classify a legal document.
        
        Args:
            text: Document text to classify
            
        Returns:
            ClassificationResult with doctrine, court level, year, and confidence scores
        """
        if self.tokenizer is None or self.doctrine_model is None or self.court_model is None:
            raise ValueError("Models not initialized. Call load_trained_models() or train models first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.classifier_config['max_length'],
            return_tensors='pt'
        )
        
        # Classify doctrine
        with torch.no_grad():
            doctrine_outputs = self.doctrine_model(**inputs)
            doctrine_probs = torch.softmax(doctrine_outputs.logits, dim=-1)
            doctrine_pred = torch.argmax(doctrine_probs, dim=-1).item()
            doctrine_confidence = doctrine_probs[0][doctrine_pred].item()
        
        # Classify court level
        with torch.no_grad():
            court_outputs = self.court_model(**inputs)
            court_probs = torch.softmax(court_outputs.logits, dim=-1)
            court_pred = torch.argmax(court_probs, dim=-1).item()
            court_confidence = court_probs[0][court_pred].item()
        
        # Extract year
        year = self._extract_year(text)
        
        # Prepare result
        result = ClassificationResult(
            doctrine=self.doctrine_id_to_label[doctrine_pred],
            court_level=self.court_id_to_label[court_pred],
            year=year,
            confidence_scores={
                'doctrine': doctrine_confidence,
                'court': court_confidence
            },
            metadata={
                'text_length': len(text),
                'has_year': year is not None,
                'doctrine_probs': doctrine_probs[0].tolist(),
                'court_probs': court_probs[0].tolist()
            }
        )
        
        return result
    
    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple documents in batch."""
        results = []
        for text in texts:
            result = self.classify_document(text)
            results.append(result)
        return results
