"""
Performance Optimization Module

This module provides performance optimization features including
batch processing, caching, and scaling for large document collections.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Iterator
import time
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from dataclasses import dataclass
from functools import lru_cache
import threading

try:
    from .utils import load_config
except ImportError:
    from utils import load_config

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    batch_size: int = 100
    max_workers: int = 4
    cache_size: int = 1000
    cache_ttl_hours: int = 24
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    memory_limit_gb: float = 4.0


class CacheManager:
    """Thread-safe cache manager with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        """Initialize cache manager."""
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
        
        logger.info(f"Cache manager initialized: max_size={max_size}, ttl={ttl_hours}h")
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        elif isinstance(data, (list, tuple)):
            return hashlib.md5(str(data).encode()).hexdigest()
        elif isinstance(data, dict):
            return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            # Remove oldest items if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
            }


class BatchProcessor:
    """Batch processor for efficient document processing."""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        """Initialize batch processor."""
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        logger.info(f"Batch processor initialized: batch_size={batch_size}, max_workers={max_workers}")
    
    def process_documents_batch(self, documents: List[Dict[str, Any]], 
                                processing_function, 
                                use_multiprocessing: bool = False) -> List[Any]:
        """Process documents in batches with parallel processing."""
        results = []
        
        # Split documents into batches
        batches = [documents[i:i + self.batch_size] 
                   for i in range(0, len(documents), self.batch_size)]
        
        logger.info(f"Processing {len(documents)} documents in {len(batches)} batches")
        
        if use_multiprocessing and len(batches) > 1:
            # Use multiprocessing for CPU-intensive tasks
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                batch_results = list(executor.map(processing_function, batches))
        else:
            # Use threading for I/O-intensive tasks
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batch_results = list(executor.map(processing_function, batches))
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def process_embeddings_batch(self, texts: List[str], 
                                 embedding_function) -> np.ndarray:
        """Process embeddings in batches for memory efficiency."""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = embedding_function(batch_texts)
            embeddings.append(batch_embeddings)
            
            logger.debug(f"Processed embedding batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
        
        return np.vstack(embeddings) if embeddings else np.array([])


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self, memory_limit_gb: float = 4.0):
        """Initialize memory optimizer."""
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        
        logger.info(f"Memory optimizer initialized: limit={memory_limit_gb}GB")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024 * 1024)
        except ImportError:
            # Fallback estimation
            return 0.0
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        current_usage = self.get_memory_usage()
        return current_usage < (self.memory_limit_bytes / (1024 * 1024 * 1024))
    
    def optimize_embeddings_storage(self, embeddings: np.ndarray) -> np.ndarray:
        """Optimize embeddings storage using appropriate data types."""
        # Convert to float32 if not already (saves memory vs float64)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Normalize embeddings for better compression
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings
    
    def chunk_large_arrays(self, array: np.ndarray, max_chunk_size: int = 10000) -> Iterator[np.ndarray]:
        """Chunk large arrays for memory-efficient processing."""
        for i in range(0, len(array), max_chunk_size):
            yield array[i:i + max_chunk_size]


class PerformanceMonitor:
    """Performance monitoring and profiling."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {
            'operation_times': {},
            'memory_usage': [],
            'cache_stats': {},
            'batch_stats': {}
        }
        
        logger.info("Performance monitor initialized")
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return OperationTimer(self, operation_name)
    
    def record_operation_time(self, operation_name: str, duration: float):
        """Record operation time."""
        if operation_name not in self.metrics['operation_times']:
            self.metrics['operation_times'][operation_name] = []
        
        self.metrics['operation_times'][operation_name].append(duration)
    
    def record_memory_usage(self):
        """Record current memory usage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
            self.metrics['memory_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'memory_gb': memory_gb
            })
        except ImportError:
            pass
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'operation_statistics': {},
            'memory_statistics': {},
            'cache_statistics': self.metrics.get('cache_stats', {}),
            'batch_statistics': self.metrics.get('batch_stats', {})
        }
        
        # Operation statistics
        for op_name, times in self.metrics['operation_times'].items():
            if times:
                report['operation_statistics'][op_name] = {
                    'count': len(times),
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times)
                }
        
        # Memory statistics
        if self.metrics['memory_usage']:
            memory_values = [m['memory_gb'] for m in self.metrics['memory_usage']]
            report['memory_statistics'] = {
                'current_memory_gb': memory_values[-1] if memory_values else 0,
                'peak_memory_gb': max(memory_values) if memory_values else 0,
                'avg_memory_gb': np.mean(memory_values) if memory_values else 0
            }
        
        return report


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        """Initialize operation timer."""
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_operation_time(self.operation_name, duration)


class OptimizedVectorDatabase:
    """Optimized vector database with performance enhancements."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize optimized vector database."""
        self.config = load_config(config_path)
        self.perf_config = self._load_performance_config()
        
        # Initialize components
        self.cache_manager = CacheManager(
            max_size=self.perf_config.cache_size,
            ttl_hours=self.perf_config.cache_ttl_hours
        ) if self.perf_config.enable_caching else None
        
        self.batch_processor = BatchProcessor(
            batch_size=self.perf_config.batch_size,
            max_workers=self.perf_config.max_workers
        )
        
        self.memory_optimizer = MemoryOptimizer(
            memory_limit_gb=self.perf_config.memory_limit_gb
        )
        
        self.performance_monitor = PerformanceMonitor()
        
        # Storage
        self.embeddings = None
        self.documents = []
        self.index_metadata = {}
        
        logger.info("Optimized Vector Database initialized")
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration."""
        perf_config = self.config.get('performance', {})
        
        return PerformanceConfig(
            batch_size=perf_config.get('batch_size', 100),
            max_workers=perf_config.get('max_workers', 4),
            cache_size=perf_config.get('cache_size', 1000),
            cache_ttl_hours=perf_config.get('cache_ttl_hours', 24),
            enable_parallel_processing=perf_config.get('enable_parallel_processing', True),
            enable_caching=perf_config.get('enable_caching', True),
            memory_limit_gb=perf_config.get('memory_limit_gb', 4.0)
        )
    
    def add_documents_optimized(self, documents: List[Dict[str, Any]], 
                                embedding_function) -> bool:
        """Add documents with performance optimizations."""
        with self.performance_monitor.time_operation('add_documents'):
            try:
                # Extract texts
                texts = [doc.get('content', doc.get('text', '')) for doc in documents]
                
                # Generate embeddings in batches
                with self.performance_monitor.time_operation('generate_embeddings'):
                    embeddings = self.batch_processor.process_embeddings_batch(
                        texts, embedding_function
                    )
                
                # Optimize embeddings storage
                embeddings = self.memory_optimizer.optimize_embeddings_storage(embeddings)
                
                # Store documents and embeddings
                if self.embeddings is None:
                    self.embeddings = embeddings
                    self.documents = documents.copy()
                else:
                    self.embeddings = np.vstack([self.embeddings, embeddings])
                    self.documents.extend(documents)
                
                # Update metadata
                self.index_metadata.update({
                    'total_documents': len(self.documents),
                    'embedding_dimension': embeddings.shape[1],
                    'last_updated': datetime.now().isoformat()
                })
                
                # Record memory usage
                self.performance_monitor.record_memory_usage()
                
                logger.info(f"Added {len(documents)} documents with optimizations")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add documents: {e}")
                return False
    
    def search_optimized(self, query_embedding: np.ndarray, 
                         top_k: int = 10,
                         use_cache: bool = True) -> List[Dict[str, Any]]:
        """Perform optimized search with caching."""
        with self.performance_monitor.time_operation('search'):
            # Generate cache key
            cache_key = None
            if use_cache and self.cache_manager:
                cache_key = self.cache_manager._generate_key(
                    (query_embedding.tobytes(), top_k)
                )
                
                # Check cache
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    logger.debug("Cache hit for search query")
                    return cached_result
            
            # Perform search
            if self.embeddings is None or len(self.embeddings) == 0:
                return []
            
            # Calculate similarities
            with self.performance_monitor.time_operation('similarity_calculation'):
                similarities = np.dot(self.embeddings, query_embedding)
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    result = {
                        'id': self.documents[idx].get('id', f'doc_{idx}'),
                        'title': self.documents[idx].get('title', ''),
                        'score': float(similarities[idx]),
                        'metadata': self.documents[idx]
                    }
                    results.append(result)
            
            # Cache results
            if cache_key and self.cache_manager:
                self.cache_manager.put(cache_key, results)
            
            return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'performance_report': self.performance_monitor.get_performance_report(),
            'index_metadata': self.index_metadata,
            'memory_stats': {
                'current_usage_gb': self.memory_optimizer.get_memory_usage(),
                'within_limits': self.memory_optimizer.check_memory_limit()
            }
        }
        
        if self.cache_manager:
            stats['cache_stats'] = self.cache_manager.get_stats()
        
        return stats
    
    def optimize_index(self) -> bool:
        """Optimize the index for better performance."""
        with self.performance_monitor.time_operation('index_optimization'):
            try:
                if self.embeddings is not None:
                    # Optimize embeddings storage
                    self.embeddings = self.memory_optimizer.optimize_embeddings_storage(
                        self.embeddings
                    )
                    
                    logger.info("Index optimization completed")
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Index optimization failed: {e}")
                return False
