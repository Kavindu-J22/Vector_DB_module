"""
Enhanced Evaluation Framework

This module provides comprehensive evaluation metrics and validation
for the Vector Database Module with advanced performance assessment.
"""

import os
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from loguru import logger

try:
    from .utils import load_config
except ImportError:
    from utils import load_config


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    map_score: float  # Mean Average Precision
    mrr_score: float  # Mean Reciprocal Rank
    ndcg_score: float  # Normalized Discounted Cumulative Gain
    recall_at_k: Dict[int, float]  # Recall@K for different K values
    precision_at_k: Dict[int, float]  # Precision@K for different K values


@dataclass
class QueryEvaluation:
    """Evaluation result for a single query."""
    query_id: str
    query_text: str
    expected_docs: List[str]
    retrieved_docs: List[str]
    relevance_scores: List[float]
    metrics: EvaluationMetrics
    execution_time: float


class EnhancedEvaluationFramework:
    """Enhanced evaluation framework for vector database performance."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the evaluation framework."""
        self.config = load_config(config_path)
        self.evaluation_config = self.config.get('evaluation', {})
        
        # Evaluation parameters
        self.k_values = self.evaluation_config.get('k_values', [1, 3, 5, 10])
        self.relevance_threshold = self.evaluation_config.get('relevance_threshold', 0.5)
        
        logger.info("Enhanced Evaluation Framework initialized")
    
    def evaluate_retrieval_system(self, 
                                  retrieval_function,
                                  test_queries: List[Dict[str, Any]],
                                  document_corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of retrieval system.
        
        Args:
            retrieval_function: Function that takes query and returns ranked results
            test_queries: List of test queries with expected results
            document_corpus: Full document corpus
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Starting evaluation with {len(test_queries)} queries")
        
        query_evaluations = []
        overall_metrics = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'map_scores': [],
            'mrr_scores': [],
            'ndcg_scores': [],
            'execution_times': []
        }
        
        for i, query_data in enumerate(test_queries):
            logger.info(f"Evaluating query {i+1}/{len(test_queries)}: {query_data['query'][:50]}...")
            
            # Evaluate single query
            query_eval = self._evaluate_single_query(
                query_data, retrieval_function, document_corpus
            )
            
            query_evaluations.append(query_eval)
            
            # Collect metrics for overall statistics
            overall_metrics['precision'].append(query_eval.metrics.precision)
            overall_metrics['recall'].append(query_eval.metrics.recall)
            overall_metrics['f1_score'].append(query_eval.metrics.f1_score)
            overall_metrics['map_scores'].append(query_eval.metrics.map_score)
            overall_metrics['mrr_scores'].append(query_eval.metrics.mrr_score)
            overall_metrics['ndcg_scores'].append(query_eval.metrics.ndcg_score)
            overall_metrics['execution_times'].append(query_eval.execution_time)
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics(overall_metrics, query_evaluations)
        
        # Generate evaluation report
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_statistics': overall_stats,
            'query_evaluations': [asdict(qe) for qe in query_evaluations],
            'configuration': self.evaluation_config,
            'summary': self._generate_summary(overall_stats)
        }
        
        logger.info("Evaluation completed successfully")
        return evaluation_report
    
    def _evaluate_single_query(self, 
                               query_data: Dict[str, Any],
                               retrieval_function,
                               document_corpus: List[Dict[str, Any]]) -> QueryEvaluation:
        """Evaluate a single query."""
        query_text = query_data['query']
        expected_docs = set(query_data['expected_docs'])
        
        # Measure execution time
        start_time = datetime.now()
        
        try:
            # Retrieve documents
            retrieved_results = retrieval_function(query_text, top_k=20)
            retrieved_docs = [result.get('id', result.get('doc_id', '')) for result in retrieved_results]
            relevance_scores = [result.get('relevance_score', result.get('score', 0.0)) for result in retrieved_results]
        except Exception as e:
            logger.error(f"Error during retrieval for query '{query_text}': {e}")
            retrieved_docs = []
            relevance_scores = []
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate metrics
        metrics = self._calculate_query_metrics(expected_docs, retrieved_docs, relevance_scores)
        
        return QueryEvaluation(
            query_id=query_data.get('id', f"query_{hash(query_text)}"),
            query_text=query_text,
            expected_docs=list(expected_docs),
            retrieved_docs=retrieved_docs,
            relevance_scores=relevance_scores,
            metrics=metrics,
            execution_time=execution_time
        )
    
    def _calculate_query_metrics(self, 
                                 expected_docs: set,
                                 retrieved_docs: List[str],
                                 relevance_scores: List[float]) -> EvaluationMetrics:
        """Calculate comprehensive metrics for a single query."""
        if not retrieved_docs:
            return EvaluationMetrics(
                precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0,
                map_score=0.0, mrr_score=0.0, ndcg_score=0.0,
                recall_at_k={k: 0.0 for k in self.k_values},
                precision_at_k={k: 0.0 for k in self.k_values}
            )
        
        # Basic precision and recall
        retrieved_set = set(retrieved_docs)
        true_positives = len(expected_docs.intersection(retrieved_set))
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(expected_docs) if expected_docs else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy (considering all documents)
        accuracy = precision  # Simplified for retrieval task
        
        # Mean Average Precision (MAP)
        map_score = self._calculate_map(expected_docs, retrieved_docs)
        
        # Mean Reciprocal Rank (MRR)
        mrr_score = self._calculate_mrr(expected_docs, retrieved_docs)
        
        # Normalized Discounted Cumulative Gain (NDCG)
        ndcg_score = self._calculate_ndcg(expected_docs, retrieved_docs, relevance_scores)
        
        # Precision@K and Recall@K
        precision_at_k = {}
        recall_at_k = {}
        
        for k in self.k_values:
            if k <= len(retrieved_docs):
                top_k_docs = set(retrieved_docs[:k])
                tp_k = len(expected_docs.intersection(top_k_docs))
                
                precision_at_k[k] = tp_k / k
                recall_at_k[k] = tp_k / len(expected_docs) if expected_docs else 0.0
            else:
                precision_at_k[k] = precision
                recall_at_k[k] = recall
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            map_score=map_score,
            mrr_score=mrr_score,
            ndcg_score=ndcg_score,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k
        )
    
    def _calculate_map(self, expected_docs: set, retrieved_docs: List[str]) -> float:
        """Calculate Mean Average Precision."""
        if not expected_docs or not retrieved_docs:
            return 0.0
        
        relevant_found = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in expected_docs:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(expected_docs) if expected_docs else 0.0
    
    def _calculate_mrr(self, expected_docs: set, retrieved_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not expected_docs or not retrieved_docs:
            return 0.0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in expected_docs:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_ndcg(self, expected_docs: set, retrieved_docs: List[str], 
                        relevance_scores: List[float], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not expected_docs or not retrieved_docs:
            return 0.0
        
        # Create relevance labels (1 for relevant, 0 for not relevant)
        relevance_labels = [1 if doc_id in expected_docs else 0 for doc_id in retrieved_docs[:k]]
        
        if not any(relevance_labels):
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_labels):
            if rel > 0:
                dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevance = sorted([1] * len(expected_docs) + [0] * (k - len(expected_docs)), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            if rel > 0:
                idcg += rel / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_overall_statistics(self, 
                                      overall_metrics: Dict[str, List[float]],
                                      query_evaluations: List[QueryEvaluation]) -> Dict[str, Any]:
        """Calculate overall statistics across all queries."""
        stats = {}
        
        # Basic statistics
        for metric_name, values in overall_metrics.items():
            if values:
                stats[f'avg_{metric_name}'] = np.mean(values)
                stats[f'std_{metric_name}'] = np.std(values)
                stats[f'min_{metric_name}'] = np.min(values)
                stats[f'max_{metric_name}'] = np.max(values)
                stats[f'median_{metric_name}'] = np.median(values)
        
        # Aggregate Precision@K and Recall@K
        for k in self.k_values:
            precision_k_values = [qe.metrics.precision_at_k[k] for qe in query_evaluations]
            recall_k_values = [qe.metrics.recall_at_k[k] for qe in query_evaluations]
            
            stats[f'avg_precision_at_{k}'] = np.mean(precision_k_values)
            stats[f'avg_recall_at_{k}'] = np.mean(recall_k_values)
        
        # Performance statistics
        stats['total_queries'] = len(query_evaluations)
        stats['avg_execution_time'] = np.mean(overall_metrics['execution_times'])
        stats['total_execution_time'] = np.sum(overall_metrics['execution_times'])
        
        return stats
    
    def _generate_summary(self, overall_stats: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable summary."""
        summary = {}
        
        # Performance summary
        avg_f1 = overall_stats.get('avg_f1_score', 0.0)
        avg_precision = overall_stats.get('avg_precision', 0.0)
        avg_recall = overall_stats.get('avg_recall', 0.0)
        
        if avg_f1 >= 0.8:
            performance_level = "Excellent"
        elif avg_f1 >= 0.6:
            performance_level = "Good"
        elif avg_f1 >= 0.4:
            performance_level = "Fair"
        else:
            performance_level = "Poor"
        
        summary['performance_level'] = performance_level
        summary['key_metrics'] = f"F1: {avg_f1:.3f}, Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}"
        
        # Speed summary
        avg_time = overall_stats.get('avg_execution_time', 0.0)
        if avg_time < 0.1:
            speed_level = "Very Fast"
        elif avg_time < 0.5:
            speed_level = "Fast"
        elif avg_time < 2.0:
            speed_level = "Moderate"
        else:
            speed_level = "Slow"
        
        summary['speed_level'] = speed_level
        summary['avg_response_time'] = f"{avg_time:.3f} seconds"
        
        # Recommendations
        recommendations = []
        if avg_precision < 0.6:
            recommendations.append("Consider improving embedding quality or ranking algorithm")
        if avg_recall < 0.6:
            recommendations.append("Consider expanding search scope or improving query processing")
        if avg_time > 1.0:
            recommendations.append("Consider optimizing search performance")
        
        summary['recommendations'] = recommendations if recommendations else ["System performance is satisfactory"]
        
        return summary
    
    def save_evaluation_report(self, evaluation_report: Dict[str, Any], 
                               output_path: str = None) -> str:
        """Save evaluation report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation_report_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to: {output_path}")
        return output_path
    
    def generate_evaluation_plots(self, evaluation_report: Dict[str, Any], 
                                  output_dir: str = "evaluation_plots") -> List[str]:
        """Generate visualization plots for evaluation results."""
        os.makedirs(output_dir, exist_ok=True)
        plot_files = []
        
        try:
            # Extract data for plotting
            query_evals = evaluation_report['query_evaluations']
            
            # Plot 1: Precision vs Recall scatter plot
            precisions = [qe['metrics']['precision'] for qe in query_evals]
            recalls = [qe['metrics']['recall'] for qe in query_evals]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(recalls, precisions, alpha=0.6)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision vs Recall by Query')
            plt.grid(True, alpha=0.3)
            
            plot_file = os.path.join(output_dir, 'precision_recall_scatter.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
            
            # Plot 2: F1 Score distribution
            f1_scores = [qe['metrics']['f1_score'] for qe in query_evals]
            
            plt.figure(figsize=(10, 6))
            plt.hist(f1_scores, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('F1 Score')
            plt.ylabel('Number of Queries')
            plt.title('F1 Score Distribution')
            plt.axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_file = os.path.join(output_dir, 'f1_score_distribution.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
            
            logger.info(f"Generated {len(plot_files)} evaluation plots in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
        
        return plot_files
