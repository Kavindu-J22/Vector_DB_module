"""
Evaluation Framework for Vector Database Module

This module provides comprehensive evaluation metrics and testing
for the retrieval system, including Recall@K, MRR, NDCG, and
comparison of filtered vs unfiltered search performance.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import json
import time
from collections import defaultdict
from loguru import logger

from .retrieval import HybridRetriever, HybridSearchResult
from .utils import load_config, save_json, load_json


@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    query_id: str
    query_text: str
    relevant_docs: Set[str]
    retrieved_docs: List[str]
    retrieved_scores: List[float]
    filters_used: Optional[Dict[str, Any]]
    search_time: float
    metrics: Dict[str, float]


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    dataset_name: str
    total_queries: int
    avg_metrics: Dict[str, float]
    per_query_results: List[QueryResult]
    filtered_vs_unfiltered: Dict[str, Dict[str, float]]
    system_stats: Dict[str, Any]
    timestamp: str


class EvaluationFramework:
    """
    Comprehensive evaluation framework for the vector database system.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize evaluation framework."""
        self.config = load_config(config_path)
        self.eval_config = self.config['evaluation']
        self.metrics_to_compute = self.eval_config['metrics']
        
        logger.info("EvaluationFramework initialized")
    
    def compute_recall_at_k(self, relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """Compute Recall@K metric."""
        if not relevant_docs:
            return 0.0
        
        retrieved_at_k = set(retrieved_docs[:k])
        relevant_retrieved = len(relevant_docs.intersection(retrieved_at_k))
        
        return relevant_retrieved / len(relevant_docs)
    
    def compute_precision_at_k(self, relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """Compute Precision@K metric."""
        if k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved_docs[:k])
        relevant_retrieved = len(relevant_docs.intersection(retrieved_at_k))
        
        return relevant_retrieved / min(k, len(retrieved_docs))
    
    def compute_mrr(self, relevant_docs: Set[str], retrieved_docs: List[str]) -> float:
        """Compute Mean Reciprocal Rank (MRR)."""
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    def compute_ndcg_at_k(self, relevant_docs: Set[str], retrieved_docs: List[str], 
                         retrieved_scores: List[float], k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain (NDCG@K)."""
        if not relevant_docs or k == 0:
            return 0.0
        
        # DCG calculation
        dcg = 0.0
        for i in range(min(k, len(retrieved_docs))):
            if retrieved_docs[i] in relevant_docs:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # IDCG calculation (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_docs))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def compute_f1_at_k(self, relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """Compute F1@K metric."""
        precision = self.compute_precision_at_k(relevant_docs, retrieved_docs, k)
        recall = self.compute_recall_at_k(relevant_docs, retrieved_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate_query(self, query_id: str, query_text: str, relevant_docs: Set[str],
                      retrieved_results: List[HybridSearchResult], 
                      filters_used: Optional[Dict[str, Any]] = None,
                      search_time: float = 0.0) -> QueryResult:
        """Evaluate a single query."""
        retrieved_docs = [result.doc_id for result in retrieved_results]
        retrieved_scores = [result.combined_score for result in retrieved_results]
        
        # Compute metrics
        metrics = {}
        
        if "recall_at_5" in self.metrics_to_compute:
            metrics["recall_at_5"] = self.compute_recall_at_k(relevant_docs, retrieved_docs, 5)
        
        if "recall_at_10" in self.metrics_to_compute:
            metrics["recall_at_10"] = self.compute_recall_at_k(relevant_docs, retrieved_docs, 10)
        
        if "precision_at_5" in self.metrics_to_compute:
            metrics["precision_at_5"] = self.compute_precision_at_k(relevant_docs, retrieved_docs, 5)
        
        if "precision_at_10" in self.metrics_to_compute:
            metrics["precision_at_10"] = self.compute_precision_at_k(relevant_docs, retrieved_docs, 10)
        
        if "mrr" in self.metrics_to_compute:
            metrics["mrr"] = self.compute_mrr(relevant_docs, retrieved_docs)
        
        if "ndcg_at_5" in self.metrics_to_compute:
            metrics["ndcg_at_5"] = self.compute_ndcg_at_k(relevant_docs, retrieved_docs, retrieved_scores, 5)
        
        if "ndcg_at_10" in self.metrics_to_compute:
            metrics["ndcg_at_10"] = self.compute_ndcg_at_k(relevant_docs, retrieved_docs, retrieved_scores, 10)
        
        if "f1_at_5" in self.metrics_to_compute:
            metrics["f1_at_5"] = self.compute_f1_at_k(relevant_docs, retrieved_docs, 5)
        
        return QueryResult(
            query_id=query_id,
            query_text=query_text,
            relevant_docs=relevant_docs,
            retrieved_docs=retrieved_docs,
            retrieved_scores=retrieved_scores,
            filters_used=filters_used,
            search_time=search_time,
            metrics=metrics
        )
    
    def evaluate_retrieval_system(self, retriever: HybridRetriever, 
                                 test_queries: List[Dict[str, Any]],
                                 compare_filtered: bool = True) -> EvaluationReport:
        """
        Evaluate the complete retrieval system.
        
        Args:
            retriever: HybridRetriever instance to evaluate
            test_queries: List of test queries with format:
                {
                    "id": "query_1",
                    "query": "contract breach damages",
                    "relevant_docs": ["doc_1", "doc_3"],
                    "filters": {"doctrine": "contract", "year": {"$gte": 2010}}
                }
            compare_filtered: Whether to compare filtered vs unfiltered search
            
        Returns:
            EvaluationReport with comprehensive results
        """
        logger.info(f"Starting evaluation with {len(test_queries)} queries")
        
        query_results = []
        filtered_results = []
        unfiltered_results = []
        
        for query_data in test_queries:
            query_id = query_data["id"]
            query_text = query_data["query"]
            relevant_docs = set(query_data["relevant_docs"])
            filters = query_data.get("filters")
            
            logger.info(f"Evaluating query: {query_id}")
            
            # Unfiltered search
            start_time = time.time()
            unfiltered_search_results = retriever.search(
                query_text, 
                top_k=10, 
                filters=None,
                use_vector=True,
                use_bm25=True
            )
            unfiltered_search_time = time.time() - start_time
            
            unfiltered_result = self.evaluate_query(
                query_id, query_text, relevant_docs, 
                unfiltered_search_results, None, unfiltered_search_time
            )
            query_results.append(unfiltered_result)
            unfiltered_results.append(unfiltered_result)
            
            # Filtered search (if filters provided and comparison requested)
            if compare_filtered and filters:
                start_time = time.time()
                filtered_search_results = retriever.search(
                    query_text,
                    top_k=10,
                    filters=filters,
                    use_vector=True,
                    use_bm25=True
                )
                filtered_search_time = time.time() - start_time
                
                filtered_result = self.evaluate_query(
                    f"{query_id}_filtered", query_text, relevant_docs,
                    filtered_search_results, filters, filtered_search_time
                )
                query_results.append(filtered_result)
                filtered_results.append(filtered_result)
        
        # Compute average metrics
        avg_metrics = self._compute_average_metrics(query_results)
        
        # Compare filtered vs unfiltered if applicable
        filtered_vs_unfiltered = {}
        if compare_filtered and filtered_results:
            filtered_vs_unfiltered = self._compare_filtered_unfiltered(
                unfiltered_results, filtered_results
            )
        
        # Get system statistics
        system_stats = retriever.get_stats()
        
        # Create evaluation report
        report = EvaluationReport(
            dataset_name="Legal Documents Evaluation",
            total_queries=len(test_queries),
            avg_metrics=avg_metrics,
            per_query_results=query_results,
            filtered_vs_unfiltered=filtered_vs_unfiltered,
            system_stats=system_stats,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        logger.info("Evaluation completed")
        return report
    
    def _compute_average_metrics(self, query_results: List[QueryResult]) -> Dict[str, float]:
        """Compute average metrics across all queries."""
        if not query_results:
            return {}
        
        metric_sums = defaultdict(float)
        metric_counts = defaultdict(int)
        
        for result in query_results:
            for metric_name, metric_value in result.metrics.items():
                metric_sums[metric_name] += metric_value
                metric_counts[metric_name] += 1
        
        avg_metrics = {}
        for metric_name in metric_sums:
            avg_metrics[metric_name] = metric_sums[metric_name] / metric_counts[metric_name]
        
        # Add average search time
        avg_search_time = sum(r.search_time for r in query_results) / len(query_results)
        avg_metrics["avg_search_time"] = avg_search_time
        
        return avg_metrics
    
    def _compare_filtered_unfiltered(self, unfiltered_results: List[QueryResult],
                                   filtered_results: List[QueryResult]) -> Dict[str, Dict[str, float]]:
        """Compare filtered vs unfiltered search performance."""
        unfiltered_avg = self._compute_average_metrics(unfiltered_results)
        filtered_avg = self._compute_average_metrics(filtered_results)
        
        comparison = {
            "unfiltered": unfiltered_avg,
            "filtered": filtered_avg,
            "improvement": {}
        }
        
        # Compute improvement percentages
        for metric in unfiltered_avg:
            if metric in filtered_avg and unfiltered_avg[metric] > 0:
                improvement = ((filtered_avg[metric] - unfiltered_avg[metric]) / 
                             unfiltered_avg[metric]) * 100
                comparison["improvement"][metric] = improvement
        
        return comparison
    
    def save_evaluation_report(self, report: EvaluationReport, filepath: str):
        """Save evaluation report to JSON file."""
        # Convert sets to lists for JSON serialization
        report_dict = asdict(report)
        
        for query_result in report_dict["per_query_results"]:
            query_result["relevant_docs"] = list(query_result["relevant_docs"])
        
        save_json(report_dict, filepath)
        logger.info(f"Evaluation report saved to {filepath}")
    
    def load_evaluation_report(self, filepath: str) -> EvaluationReport:
        """Load evaluation report from JSON file."""
        report_dict = load_json(filepath)
        
        # Convert lists back to sets
        for query_result in report_dict["per_query_results"]:
            query_result["relevant_docs"] = set(query_result["relevant_docs"])
        
        # Reconstruct QueryResult objects
        query_results = []
        for qr_dict in report_dict["per_query_results"]:
            query_result = QueryResult(**qr_dict)
            query_results.append(query_result)
        
        report_dict["per_query_results"] = query_results
        
        report = EvaluationReport(**report_dict)
        logger.info(f"Evaluation report loaded from {filepath}")
        return report
    
    def print_evaluation_summary(self, report: EvaluationReport):
        """Print a summary of the evaluation results."""
        print("\n" + "="*60)
        print("VECTOR DATABASE EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Dataset: {report.dataset_name}")
        print(f"Total Queries: {report.total_queries}")
        print(f"Timestamp: {report.timestamp}")
        
        print("\nAVERAGE METRICS:")
        print("-" * 30)
        for metric, value in report.avg_metrics.items():
            if metric == "avg_search_time":
                print(f"{metric:20}: {value:.4f} seconds")
            else:
                print(f"{metric:20}: {value:.4f}")
        
        if report.filtered_vs_unfiltered:
            print("\nFILTERED vs UNFILTERED COMPARISON:")
            print("-" * 40)
            
            comparison = report.filtered_vs_unfiltered
            
            print("Unfiltered Performance:")
            for metric, value in comparison["unfiltered"].items():
                if metric == "avg_search_time":
                    print(f"  {metric:18}: {value:.4f} seconds")
                else:
                    print(f"  {metric:18}: {value:.4f}")
            
            print("\nFiltered Performance:")
            for metric, value in comparison["filtered"].items():
                if metric == "avg_search_time":
                    print(f"  {metric:18}: {value:.4f} seconds")
                else:
                    print(f"  {metric:18}: {value:.4f}")
            
            print("\nImprovement (%):")
            for metric, improvement in comparison["improvement"].items():
                print(f"  {metric:18}: {improvement:+.2f}%")
        
        print("\nSYSTEM STATISTICS:")
        print("-" * 20)
        for key, value in report.system_stats.items():
            print(f"{key:20}: {value}")
        
        print("="*60)
