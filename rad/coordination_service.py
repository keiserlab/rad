#!/usr/bin/env python3
"""
Coordination Service for Distributed RAD Traversal

This module provides centralized coordination for distributed RAD workers,
managing work distribution, state synchronization, and termination conditions
across multiple compute nodes in HPC and cloud environments.

Key Features:
- Redis-based distributed state management
- Work distribution and load balancing
- Worker registration and heartbeat monitoring
- Termination condition handling (timeout, n_to_score, queue empty)
- Fault tolerance and error recovery
"""

import time
import uuid
import json
import threading
import multiprocessing
from typing import Dict, List, Tuple, Optional, Any, Set
from abc import ABC, abstractmethod
import logging

import redis

from .priority_queue import RedisPQ
from .visited import RedisVisited
from .scored import RedisScoredSet

logger = logging.getLogger(__name__)

class WorkItem:
    """Represents a work item for RAD traversal."""
    
    def __init__(self, node_id: int, level: int, score: float, 
                 request_id: Optional[str] = None):
        self.node_id = node_id
        self.level = level
        self.score = score
        self.request_id = request_id or str(uuid.uuid4())
        self.created_at = time.time()
        self.assigned_at = None
        self.assigned_to = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'level': self.level,
            'score': self.score,
            'request_id': self.request_id,
            'created_at': self.created_at,
            'assigned_at': self.assigned_at,
            'assigned_to': self.assigned_to
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkItem':
        """Create WorkItem from dictionary."""
        item = cls(
            node_id=data['node_id'],
            level=data['level'], 
            score=data['score'],
            request_id=data['request_id']
        )
        item.created_at = data['created_at']
        item.assigned_at = data.get('assigned_at')
        item.assigned_to = data.get('assigned_to')
        return item

class WorkerInfo:
    """Information about a registered worker."""
    
    def __init__(self, worker_id: str, worker_type: str = "default", 
                 capabilities: Optional[Dict] = None):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.capabilities = capabilities or {}
        self.registered_at = time.time()
        self.last_heartbeat = time.time()
        self.assigned_work = set()
        self.completed_work = 0
        self.error_count = 0
        self.status = "active"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'worker_id': self.worker_id,
            'worker_type': self.worker_type,
            'capabilities': self.capabilities,
            'registered_at': self.registered_at,
            'last_heartbeat': self.last_heartbeat,
            'assigned_work': list(self.assigned_work),
            'completed_work': self.completed_work,
            'error_count': self.error_count,
            'status': self.status
        }

class CoordinationService:
    """
    Centralized coordination service for distributed RAD traversal.
    
    This service manages:
    - Work distribution across multiple workers
    - Redis-based state synchronization
    - Worker registration and health monitoring
    - Termination condition evaluation
    - Result collection and aggregation
    """
    
    def __init__(self, redis_client: redis.Redis, 
                 namespace: str = "rad_coordination",
                 worker_timeout: float = 60.0,
                 heartbeat_interval: float = 10.0):
        """
        Initialize coordination service.
        
        Args:
            redis_client: Redis client for state management
            namespace: Redis key namespace for this coordination instance
            worker_timeout: Time before considering worker dead (seconds)
            heartbeat_interval: Expected heartbeat interval from workers
        """
        self.redis = redis_client
        self.namespace = namespace
        self.worker_timeout = worker_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # Coordination state
        self.coordination_id = str(uuid.uuid4())
        self.started_at = time.time()
        self.is_running = False
        
        # Worker management
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_lock = threading.Lock()
        
        # Termination conditions
        self.termination_conditions = {}
        self.should_terminate = False
        self.termination_reason = None
        
        # Redis-based data structures
        self.priority_queue = RedisPQ(
            redis_client=redis_client,
            queue_name=f"{namespace}:priority_queue"
        )
        self.visited_set = RedisVisited(
            redis_client=redis_client,
            visited_name=f"{namespace}:visited"
        )
        self.scored_set = RedisScoredSet(
            redis_client=redis_client,
            scored_name=f"{namespace}:scored"
        )
        
        # Redis keys for coordination
        self.keys = {
            'coordination_info': f"{namespace}:coordination_info",
            'workers': f"{namespace}:workers",
            'work_assignments': f"{namespace}:work_assignments",
            'worker_heartbeats': f"{namespace}:worker_heartbeats",
            'termination_status': f"{namespace}:termination_status"
        }
        
        # Background threads
        self.monitor_thread = None
        self.cleanup_thread = None
        
        logger.info(f"CoordinationService initialized with ID: {self.coordination_id}")
    
    def start(self, termination_conditions: Dict[str, Any]) -> None:
        """
        Start the coordination service.
        
        Args:
            termination_conditions: Dictionary with termination criteria
                - 'timeout': Maximum runtime in seconds
                - 'n_to_score': Maximum number of molecules to score
                - 'manual_stop': Allow manual termination
        """
        if self.is_running:
            raise RuntimeError("Coordination service is already running")
        
        self.termination_conditions = termination_conditions
        self.should_terminate = False
        self.termination_reason = None
        self.is_running = True
        
        # Store coordination info in Redis (serialize complex objects)
        coordination_info = {
            'coordination_id': self.coordination_id,
            'started_at': str(self.started_at),
            'termination_conditions': json.dumps(termination_conditions),
            'status': 'running'
        }
        # Convert all values to strings for Redis
        for key, value in coordination_info.items():
            self.redis.hset(self.keys['coordination_info'], key, str(value))
        
        # Start background monitoring threads
        self.monitor_thread = threading.Thread(
            target=self._monitor_workers_and_termination,
            daemon=True,
            name="CoordinationMonitor"
        )
        self.monitor_thread.start()
        
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_stale_assignments,
            daemon=True,
            name="CoordinationCleanup"
        )
        self.cleanup_thread.start()
        
        logger.info("Coordination service started")
    
    def register_worker(self, worker_id: str, worker_type: str = "default",
                       capabilities: Optional[Dict] = None) -> bool:
        """
        Register a new worker with the coordination service.
        
        Args:
            worker_id: Unique identifier for the worker
            worker_type: Type of worker (for future load balancing)
            capabilities: Worker capabilities and configuration
            
        Returns:
            True if registration successful, False if worker already exists
        """
        with self.worker_lock:
            if worker_id in self.workers:
                logger.warning(f"Worker {worker_id} already registered")
                return False
            
            worker_info = WorkerInfo(worker_id, worker_type, capabilities)
            self.workers[worker_id] = worker_info
            
            # Store in Redis
            self.redis.hset(
                self.keys['workers'],
                worker_id,
                json.dumps(worker_info.to_dict())
            )
            
            logger.info(f"Registered worker: {worker_id} (type: {worker_type})")
            return True
    
    def worker_heartbeat(self, worker_id: str) -> bool:
        """
        Process heartbeat from a worker.
        
        Args:
            worker_id: ID of the worker sending heartbeat
            
        Returns:
            True if heartbeat accepted, False if worker not registered
        """
        with self.worker_lock:
            if worker_id not in self.workers:
                logger.warning(f"Heartbeat from unregistered worker: {worker_id}")
                return False
            
            self.workers[worker_id].last_heartbeat = time.time()
            self.workers[worker_id].status = "active"
            
            # Update Redis
            self.redis.hset(
                self.keys['worker_heartbeats'],
                worker_id,
                str(time.time())
            )
            
            return True
    
    def request_work(self, worker_id: str) -> Optional[WorkItem]:
        """
        Request work for a worker.
        
        Args:
            worker_id: ID of the worker requesting work
            
        Returns:
            WorkItem if work available, None if no work or service terminating
        """
        if self.should_terminate:
            return None
        
        if worker_id not in self.workers:
            logger.warning(f"Work request from unregistered worker: {worker_id}")
            return None
        
        # Try to get work from priority queue
        work_tuple = self.priority_queue.pop()
        if work_tuple is None:
            return None
        
        node_id, level, score = work_tuple
        work_item = WorkItem(node_id, level, score)
        work_item.assigned_at = time.time()
        work_item.assigned_to = worker_id
        
        # Track assignment
        with self.worker_lock:
            self.workers[worker_id].assigned_work.add(work_item.request_id)
        
        # Store assignment in Redis
        self.redis.hset(
            self.keys['work_assignments'],
            work_item.request_id,
            json.dumps(work_item.to_dict())
        )
        
        logger.debug(f"Assigned work {work_item.request_id} to worker {worker_id}")
        return work_item
    
    def submit_work_results(self, worker_id: str, work_item: WorkItem,
                           neighbors: List, new_scores: Dict[int, tuple]) -> bool:
        """
        Submit results from completed work.
        
        Args:
            worker_id: ID of the worker submitting results
            work_item: The completed work item
            neighbors: List of neighbor data in [node_id, smiles, ...] format
            new_scores: Dictionary mapping node_id -> (score, smiles) tuples
            
        Returns:
            True if results accepted, False otherwise
        """
        if worker_id not in self.workers:
            logger.warning(f"Results from unregistered worker: {worker_id}")
            return False
        
        try:
            # Process neighbors and add new work
            for i in range(0, len(neighbors), 2):
                neighbor_id, neighbor_smiles = neighbors[i], neighbors[i+1]
                
                # Check if already visited
                if self.visited_set.checkAndInsert(neighbor_id, work_item.level):
                    continue
                
                # Get or calculate score
                if neighbor_id in new_scores:
                    score, smiles = new_scores[neighbor_id]
                    self.scored_set.insert(neighbor_id, score, smiles)  # Store with SMILES
                else:
                    # Score should have been calculated by worker
                    existing_score = self.scored_set.getScore(neighbor_id)
                    if existing_score is None:
                        logger.warning(f"No score provided for neighbor {neighbor_id}")
                        continue
                    score = existing_score
                
                # Add to priority queue
                self.priority_queue.insert(neighbor_id, work_item.level, score)
            
            # Add current node to next level if applicable
            if work_item.level > 0:
                next_level = work_item.level - 1
                if not self.visited_set.checkAndInsert(work_item.node_id, next_level):
                    self.priority_queue.insert(work_item.node_id, next_level, work_item.score)
            
            # Update worker stats
            with self.worker_lock:
                worker = self.workers[worker_id]
                worker.assigned_work.discard(work_item.request_id)
                worker.completed_work += 1
            
            # Clean up assignment tracking
            self.redis.hdel(self.keys['work_assignments'], work_item.request_id)
            
            logger.debug(f"Processed results from worker {worker_id} for work {work_item.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing results from worker {worker_id}: {e}")
            with self.worker_lock:
                self.workers[worker_id].error_count += 1
            return False
    
    def check_termination(self) -> Tuple[bool, Optional[str]]:
        """
        Check if termination conditions are met.
        
        Returns:
            Tuple of (should_terminate, reason)
        """
        if self.should_terminate:
            return True, self.termination_reason
        
        conditions = self.termination_conditions
        
        # Check timeout
        if 'timeout' in conditions:
            runtime = time.time() - self.started_at
            if runtime >= conditions['timeout']:
                return True, f"Timeout reached ({runtime:.1f}s >= {conditions['timeout']}s)"
        
        # Check number scored
        if 'n_to_score' in conditions:
            scored_count = len(self.scored_set)
            if scored_count >= conditions['n_to_score']:
                return True, f"Target molecules scored ({scored_count} >= {conditions['n_to_score']})"
        
        # Check if queue is empty and no active work
        try:
            work_tuple = self.priority_queue.pop()
            if work_tuple is None:
                # Check if any workers have assigned work
                active_assignments = sum(
                    len(worker.assigned_work) for worker in self.workers.values()
                )
                if active_assignments == 0:
                    return True, "No more work available and no active assignments"
            else:
                # Put work back
                node_id, level, score = work_tuple
                self.priority_queue.insert(node_id, level, score)
        except Exception as e:
            logger.debug(f"Error checking queue status: {e}")
            # Continue without terminating due to queue check error
        
        return False, None
    
    def shutdown(self, reason: str = "Manual shutdown") -> None:
        """
        Shutdown the coordination service.
        
        Args:
            reason: Reason for shutdown
        """
        logger.info(f"Shutting down coordination service: {reason}")
        
        self.should_terminate = True
        self.termination_reason = reason
        self.is_running = False
        
        # Update Redis with termination status (serialize complex objects)
        termination_info = {
            'terminated_at': str(time.time()),
            'reason': reason,
            'final_stats': json.dumps(self.get_coordination_stats())
        }
        # Convert all values to strings for Redis
        for key, value in termination_info.items():
            self.redis.hset(self.keys['termination_status'], key, str(value))
        
        # Wait for background threads
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=2.0)
        
        logger.info("Coordination service shutdown complete")
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination service statistics."""
        runtime = time.time() - self.started_at
        
        worker_stats = {
            'total_workers': len(self.workers),
            'active_workers': sum(1 for w in self.workers.values() if w.status == 'active'),
            'total_completed_work': sum(w.completed_work for w in self.workers.values()),
            'total_errors': sum(w.error_count for w in self.workers.values())
        }
        
        return {
            'coordination_id': self.coordination_id,
            'runtime_seconds': runtime,
            'is_running': self.is_running,
            'should_terminate': self.should_terminate,
            'termination_reason': self.termination_reason,
            'scored_molecules': len(self.scored_set),
            'pending_work': self.priority_queue.r.zcard(self.priority_queue.queue_name),
            'workers': worker_stats,
            'termination_conditions': self.termination_conditions
        }
    
    def _monitor_workers_and_termination(self):
        """Background thread that monitors workers and termination conditions."""
        while self.is_running and not self.should_terminate:
            try:
                # Check for dead workers
                current_time = time.time()
                dead_workers = []
                
                with self.worker_lock:
                    for worker_id, worker in self.workers.items():
                        if current_time - worker.last_heartbeat > self.worker_timeout:
                            if worker.status != 'dead':
                                worker.status = 'dead'
                                dead_workers.append(worker_id)
                                logger.warning(f"Worker {worker_id} marked as dead")
                
                # Reassign work from dead workers
                for worker_id in dead_workers:
                    self._reassign_worker_assignments(worker_id)
                
                # Check termination conditions
                should_terminate, reason = self.check_termination()
                if should_terminate:
                    self.shutdown(reason)
                    break
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in coordination monitor: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _cleanup_stale_assignments(self):
        """Background thread that cleans up stale work assignments."""
        while self.is_running and not self.should_terminate:
            try:
                # Clean up assignments older than 2x worker timeout
                cleanup_threshold = time.time() - (2 * self.worker_timeout)
                
                assignment_keys = self.redis.hkeys(self.keys['work_assignments'])
                for assignment_key in assignment_keys:
                    assignment_data = self.redis.hget(self.keys['work_assignments'], assignment_key)
                    if assignment_data:
                        # Decode bytes to string if necessary
                        if isinstance(assignment_data, bytes):
                            assignment_data = assignment_data.decode('utf-8')
                        assignment = json.loads(assignment_data)
                        if assignment.get('assigned_at', 0) < cleanup_threshold:
                            # Reassign this work
                            work_item = WorkItem.from_dict(assignment)
                            self.priority_queue.insert(work_item.node_id, work_item.level, work_item.score)
                            self.redis.hdel(self.keys['work_assignments'], assignment_key)
                            logger.info(f"Reassigned stale work: {assignment_key}")
                
                time.sleep(self.worker_timeout)
                
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
                time.sleep(self.worker_timeout)
    
    def _reassign_worker_assignments(self, worker_id: str):
        """Reassign work from a dead worker back to the priority queue."""
        with self.worker_lock:
            worker = self.workers.get(worker_id)
            if not worker:
                return
            
            for request_id in list(worker.assigned_work):
                assignment_data = self.redis.hget(self.keys['work_assignments'], request_id)
                if assignment_data:
                    # Decode bytes to string if necessary
                    if isinstance(assignment_data, bytes):
                        assignment_data = assignment_data.decode('utf-8')
                    assignment = json.loads(assignment_data)
                    work_item = WorkItem.from_dict(assignment)
                    
                    # Put work back in queue
                    self.priority_queue.insert(work_item.node_id, work_item.level, work_item.score)
                    
                    # Clean up assignment
                    self.redis.hdel(self.keys['work_assignments'], request_id)
                    worker.assigned_work.discard(request_id)
                    
                    logger.info(f"Reassigned work {request_id} from dead worker {worker_id}")


def create_coordination_service(redis_client: redis.Redis, **kwargs) -> CoordinationService:
    """
    Convenience function to create a coordination service.
    
    Args:
        redis_client: Redis client for state management
        **kwargs: Additional arguments for CoordinationService
        
    Returns:
        Configured CoordinationService instance
    """
    return CoordinationService(redis_client, **kwargs)