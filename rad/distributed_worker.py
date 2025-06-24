#!/usr/bin/env python3
"""
Distributed Worker for Scalable RAD Architecture

This module implements lightweight, network-aware workers that can run on any
compute node and connect to remote HNSW and coordination services. Workers are
designed for HPC clusters, cloud environments, and hybrid deployments.

Key Features:
- Service discovery for HNSW and coordination services
- Robust error handling and reconnection logic
- Configurable work processing and scoring
- Performance monitoring and health reporting
- Graceful shutdown and cleanup
"""

import time
import uuid
import threading
import signal
import sys
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging

from .hnsw_service import HNSWService, service_registry
from .coordination_service import CoordinationService, WorkItem

logger = logging.getLogger(__name__)

class DistributedWorker:
    """
    Lightweight, distributed worker for RAD traversal.
    
    Workers can run on any compute node and connect to:
    - HNSW Service (local process or remote API)
    - Coordination Service (Redis-based)
    
    Workers are designed to be fault-tolerant and can handle:
    - Network interruptions
    - Service restarts
    - Load balancing
    - Graceful shutdown
    """
    
    def __init__(self, worker_id: Optional[str] = None,
                 hnsw_service: Optional[HNSWService] = None,
                 coordination_service: Optional[CoordinationService] = None,
                 scoring_fn: Optional[Callable] = None,
                 worker_type: str = "default",
                 capabilities: Optional[Dict] = None,
                 heartbeat_interval: float = 10.0,
                 work_timeout: float = 30.0,
                 max_retries: int = 3):
        """
        Initialize distributed worker.
        
        Args:
            worker_id: Unique worker identifier (auto-generated if None)
            hnsw_service: HNSW service for neighbor queries
            coordination_service: Coordination service for work management
            scoring_fn: Function to score molecules
            worker_type: Type of worker for load balancing
            capabilities: Worker capabilities and configuration
            heartbeat_interval: Interval for sending heartbeats (seconds)
            work_timeout: Timeout for processing work items (seconds)
            max_retries: Maximum retries for failed operations
        """
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.worker_type = worker_type
        self.capabilities = capabilities or {}
        self.heartbeat_interval = heartbeat_interval
        self.work_timeout = work_timeout
        self.max_retries = max_retries
        
        # Services
        self.hnsw_service = hnsw_service
        self.coordination_service = coordination_service
        self.scoring_fn = scoring_fn
        
        # Worker state
        self.is_running = False
        self.should_stop = False
        self.started_at = None
        self.last_work_at = None
        
        # Statistics
        self.work_completed = 0
        self.work_failed = 0
        self.total_score_time = 0.0
        self.total_neighbor_time = 0.0
        self.errors = []
        
        # Threading
        self.heartbeat_thread = None
        self.work_thread = None
        self.worker_lock = threading.Lock()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"DistributedWorker initialized: {self.worker_id}")
    
    def connect_services(self, hnsw_service_name: Optional[str] = None,
                        hnsw_service: Optional[HNSWService] = None,
                        coordination_service: Optional[CoordinationService] = None) -> bool:
        """
        Connect to HNSW and coordination services.
        
        Args:
            hnsw_service_name: Name of HNSW service in registry
            hnsw_service: Direct HNSW service instance
            coordination_service: Coordination service instance
            
        Returns:
            True if all services connected successfully
        """
        try:
            # Connect to HNSW service
            if hnsw_service:
                self.hnsw_service = hnsw_service
            elif hnsw_service_name:
                self.hnsw_service = service_registry.get_service(hnsw_service_name)
            elif not self.hnsw_service:
                # Try to get default service
                self.hnsw_service = service_registry.get_service()
            
            if not self.hnsw_service:
                logger.error("No HNSW service available")
                return False
            
            # Test HNSW service
            if not self.hnsw_service.is_healthy():
                logger.error("HNSW service is not healthy")
                return False
            
            # Connect to coordination service
            if coordination_service:
                self.coordination_service = coordination_service
            
            if not self.coordination_service:
                logger.error("No coordination service provided")
                return False
            
            logger.info(f"Worker {self.worker_id} connected to services")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to services: {e}")
            return False
    
    def start(self, register_worker: bool = True) -> bool:
        """
        Start the distributed worker.
        
        Args:
            register_worker: Whether to register with coordination service
            
        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning(f"Worker {self.worker_id} is already running")
            return False
        
        if not self.hnsw_service or not self.coordination_service:
            logger.error("Services not connected. Call connect_services() first.")
            return False
        
        if not self.scoring_fn:
            logger.error("No scoring function provided")
            return False
        
        try:
            # Register with coordination service
            if register_worker:
                success = self.coordination_service.register_worker(
                    self.worker_id, self.worker_type, self.capabilities
                )
                if not success:
                    logger.error(f"Failed to register worker {self.worker_id}")
                    return False
            
            # Start worker
            self.is_running = True
            self.should_stop = False
            self.started_at = time.time()
            
            # Start background threads
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name=f"Worker-{self.worker_id}-Heartbeat"
            )
            self.heartbeat_thread.start()
            
            self.work_thread = threading.Thread(
                target=self._work_loop,
                daemon=True,
                name=f"Worker-{self.worker_id}-Work"
            )
            self.work_thread.start()
            
            logger.info(f"Worker {self.worker_id} started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting worker {self.worker_id}: {e}")
            self.is_running = False
            return False
    
    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the distributed worker gracefully.
        
        Args:
            timeout: Maximum time to wait for shutdown (seconds)
        """
        if not self.is_running:
            return
        
        logger.info(f"Stopping worker {self.worker_id}")
        self.should_stop = True
        self.is_running = False
        
        # Wait for threads to finish
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=timeout/2)
        
        if self.work_thread and self.work_thread.is_alive():
            self.work_thread.join(timeout=timeout/2)
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics and status."""
        runtime = time.time() - self.started_at if self.started_at else 0
        
        return {
            'worker_id': self.worker_id,
            'worker_type': self.worker_type,
            'capabilities': self.capabilities,
            'is_running': self.is_running,
            'runtime_seconds': runtime,
            'work_completed': self.work_completed,
            'work_failed': self.work_failed,
            'success_rate': self.work_completed / max(self.work_completed + self.work_failed, 1),
            'avg_score_time': self.total_score_time / max(self.work_completed, 1),
            'avg_neighbor_time': self.total_neighbor_time / max(self.work_completed, 1),
            'last_work_at': self.last_work_at,
            'error_count': len(self.errors),
            'recent_errors': self.errors[-5:] if self.errors else []
        }
    
    def _heartbeat_loop(self):
        """Background thread that sends heartbeats to coordination service."""
        while self.is_running and not self.should_stop:
            try:
                success = self.coordination_service.worker_heartbeat(self.worker_id)
                if not success:
                    logger.warning(f"Heartbeat rejected for worker {self.worker_id}")
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                self._record_error(f"Heartbeat error: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _work_loop(self):
        """Main work processing loop."""
        while self.is_running and not self.should_stop:
            try:
                # Request work from coordination service
                work_item = self.coordination_service.request_work(self.worker_id)
                
                if work_item is None:
                    # No work available, wait and retry
                    time.sleep(1.0)
                    continue
                
                # Process the work item
                success = self._process_work_item(work_item)
                
                if success:
                    with self.worker_lock:
                        self.work_completed += 1
                        self.last_work_at = time.time()
                else:
                    with self.worker_lock:
                        self.work_failed += 1
                
            except Exception as e:
                logger.error(f"Error in work loop: {e}")
                self._record_error(f"Work loop error: {e}")
                time.sleep(1.0)
    
    def _process_work_item(self, work_item: WorkItem) -> bool:
        """
        Process a single work item.
        
        Args:
            work_item: Work item to process
            
        Returns:
            True if processed successfully
        """
        try:
            start_time = time.time()
            
            # Get neighbors from HNSW service
            neighbor_start = time.time()
            neighbors = self.hnsw_service.get_neighbors(work_item.node_id, work_item.level)
            neighbor_time = time.time() - neighbor_start
            
            # Score the neighbors
            score_start = time.time()
            new_scores = {}
            
            for i in range(0, len(neighbors), 2):
                neighbor_id, neighbor_key = neighbors[i], neighbors[i+1]
                
                try:
                    # Check if we already have a score
                    existing_score = self.coordination_service.scored_set.getScore(neighbor_key)
                    if existing_score is None:
                        # Calculate new score
                        score = self.scoring_fn(neighbor_key)
                        new_scores[neighbor_key] = score
                except Exception as e:
                    logger.warning(f"Error scoring neighbor {neighbor_key}: {e}")
                    continue
            
            score_time = time.time() - score_start
            
            # Submit results to coordination service
            success = self.coordination_service.submit_work_results(
                self.worker_id, work_item, neighbors, new_scores
            )
            
            if success:
                # Update timing statistics
                with self.worker_lock:
                    self.total_neighbor_time += neighbor_time
                    self.total_score_time += score_time
                
                total_time = time.time() - start_time
                logger.debug(f"Processed work {work_item.request_id} in {total_time:.3f}s "
                           f"(neighbors: {neighbor_time:.3f}s, scoring: {score_time:.3f}s)")
                return True
            else:
                logger.error(f"Failed to submit results for work {work_item.request_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing work item {work_item.request_id}: {e}")
            self._record_error(f"Work processing error: {e}")
            return False
    
    def _record_error(self, error_msg: str):
        """Record an error with timestamp."""
        error_record = {
            'timestamp': time.time(),
            'message': error_msg
        }
        
        with self.worker_lock:
            self.errors.append(error_record)
            # Keep only last 100 errors
            if len(self.errors) > 100:
                self.errors = self.errors[-100:]
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Worker {self.worker_id} received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)


class WorkerPool:
    """
    Pool of distributed workers for coordinated execution.
    
    This class manages multiple workers on a single compute node,
    providing easy scaling and management.
    """
    
    def __init__(self, num_workers: int, worker_config: Dict[str, Any]):
        """
        Initialize worker pool.
        
        Args:
            num_workers: Number of workers to create
            worker_config: Configuration for each worker
        """
        self.num_workers = num_workers
        self.worker_config = worker_config
        self.workers: List[DistributedWorker] = []
        self.is_running = False
    
    def start_all(self) -> bool:
        """Start all workers in the pool."""
        if self.is_running:
            return False
        
        try:
            # Create and start workers
            for i in range(self.num_workers):
                worker_id_prefix = self.worker_config.get('worker_id_prefix', 'worker')
                worker_id = f"{worker_id_prefix}_{i}"
                
                # Filter out worker_id_prefix from config before passing to DistributedWorker
                worker_config = {k: v for k, v in self.worker_config.items() if k != 'worker_id_prefix'}
                
                worker = DistributedWorker(
                    worker_id=worker_id,
                    **worker_config
                )
                
                if not worker.connect_services():
                    logger.error(f"Failed to connect services for worker {worker_id}")
                    return False
                
                if not worker.start():
                    logger.error(f"Failed to start worker {worker_id}")
                    return False
                
                self.workers.append(worker)
            
            self.is_running = True
            logger.info(f"Started {len(self.workers)} workers")
            return True
            
        except Exception as e:
            logger.error(f"Error starting worker pool: {e}")
            self.stop_all()
            return False
    
    def stop_all(self, timeout: float = 10.0) -> None:
        """Stop all workers in the pool."""
        if not self.is_running:
            return
        
        logger.info(f"Stopping {len(self.workers)} workers")
        
        for worker in self.workers:
            try:
                worker.stop(timeout=timeout/len(self.workers))
            except Exception as e:
                logger.error(f"Error stopping worker {worker.worker_id}: {e}")
        
        self.workers.clear()
        self.is_running = False
        logger.info("Worker pool stopped")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics for all workers in the pool."""
        if not self.workers:
            return {'num_workers': 0, 'workers': []}
        
        worker_stats = [worker.get_worker_stats() for worker in self.workers]
        
        return {
            'num_workers': len(self.workers),
            'is_running': self.is_running,
            'total_work_completed': sum(w['work_completed'] for w in worker_stats),
            'total_work_failed': sum(w['work_failed'] for w in worker_stats),
            'avg_success_rate': sum(w['success_rate'] for w in worker_stats) / len(worker_stats),
            'workers': worker_stats
        }


def create_worker_pool(num_workers: int, **worker_config) -> WorkerPool:
    """
    Convenience function to create a worker pool.
    
    Args:
        num_workers: Number of workers to create
        **worker_config: Configuration for each worker
        
    Returns:
        Configured WorkerPool instance
    """
    return WorkerPool(num_workers, worker_config)