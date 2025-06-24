#!/usr/bin/env python3
"""
Scalable RAD Traverser with Service-Oriented Architecture

This is a complete rewrite of the RAD traverser using the new scalable architecture
that supports local deployment, HPC clusters, and cloud environments.

Key Features:
- Service-oriented design with pluggable HNSW and coordination services
- Support for distributed workers across multiple compute nodes
- Flexible deployment modes (local, distributed, hybrid)
- Robust error handling and monitoring
- Backward-compatible API for easy migration
"""

from typing import Union, Optional, Dict, Any, Callable
import redis
import time
import logging

from .hnsw_service import HNSWService, LocalHNSWService, create_local_hnsw_service
from .coordination_service import CoordinationService, create_coordination_service
from .distributed_worker import DistributedWorker, WorkerPool, create_worker_pool
from .redis_server import RedisServer

logger = logging.getLogger(__name__)

class RADTraverser:
    """
    Scalable RAD Traverser with service-oriented architecture.
    
    This traverser supports multiple deployment modes:
    - Local: All services on single machine (like old implementation)
    - Distributed: Services on separate nodes (HPC clusters)
    - Hybrid: Mix of local and remote services (cloud integration)
    
    The API maintains compatibility with the original implementation while
    providing new capabilities for scalable deployment.
    """
    
    def __init__(self, 
                 hnsw,
                 scoring_fn: Callable,
                 deployment_mode: str = "local",
                 redis_host: Optional[str] = None,
                 redis_port: int = 6379,
                 namespace: Optional[str] = None,
                 **kwargs):
        """
        Initialize RAD traverser with scalable architecture.
        
        Args:
            hnsw: HNSW index for neighbor queries
            scoring_fn: Function to score molecules
            deployment_mode: "local", "distributed", or "hybrid"
            redis_host: Redis host (None for local Redis)
            redis_port: Redis port
            namespace: Namespace for this traversal session
            **kwargs: Additional configuration options
        """
        self.hnsw = hnsw
        self.scoring_fn = scoring_fn
        self.deployment_mode = deployment_mode
        self.namespace = namespace or f"rad_session_{int(time.time())}"
        
        # Service instances
        self.hnsw_service: Optional[HNSWService] = None
        self.coordination_service: Optional[CoordinationService] = None
        self.redis_client: Optional[redis.Redis] = None
        self.redis_server: Optional[RedisServer] = None
        
        # Worker management
        self.workers: list = []
        self.worker_pool: Optional[WorkerPool] = None
        
        # State
        self.is_initialized = False
        self.is_running = False
        
        # Initialize services based on deployment mode
        self._init_services(redis_host, redis_port, **kwargs)
        
        logger.info(f"RADTraverser initialized in {deployment_mode} mode")
    
    def _init_services(self, redis_host: Optional[str], redis_port: int, **kwargs):
        """Initialize services based on deployment mode."""
        try:
            # Initialize Redis client
            if redis_host is not None:
                logger.info(f'Connecting to established redis server at {redis_host}:{redis_port}')
                self.redis_client = redis.StrictRedis(
                    host=redis_host, 
                    port=redis_port,
                    decode_responses=False
                )
            else:
                logger.info(f'Starting local redis server on port {redis_port}')
                self.redis_server = RedisServer(redis_port=redis_port, **kwargs)
                self.redis_client = self.redis_server.getClient()
            
            # Test Redis connection
            self.redis_client.ping()
            
            # Initialize HNSW service
            if self.deployment_mode == "local":
                self.hnsw_service = create_local_hnsw_service(self.hnsw, **kwargs)
            elif self.deployment_mode == "distributed":
                # For distributed mode, assume HNSW service is provided externally
                # or we'll connect to it later
                self.hnsw_service = LocalHNSWService(self.hnsw, **kwargs)
            else:
                raise ValueError(f"Unsupported deployment mode: {self.deployment_mode}")
            
            # Initialize coordination service
            self.coordination_service = create_coordination_service(
                self.redis_client,
                namespace=self.namespace,
                **kwargs
            )
            
            self.is_initialized = True
            logger.info("Services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            raise
    
    def prime(self, **kwargs):
        """
        Prime the traversal by scoring top-level nodes and adding them to the priority queue.
        
        Args:
            **kwargs: Additional arguments passed to scoring function
        """
        if not self.is_initialized:
            raise RuntimeError("Services not initialized")
        
        logger.info("Priming traversal with top-level nodes...")
        
        try:
            # Get top level nodes from HNSW
            top_level_nodes = self.hnsw_service.get_top_level_nodes()
            
            scored_count = 0
            for i in range(0, len(top_level_nodes), 2):
                node_id, node_key = top_level_nodes[i], top_level_nodes[i+1]
                
                # Score the node
                score = self.scoring_fn(node_key, **kwargs)
                
                # Add to scored set
                self.coordination_service.scored_set.insert(key=node_key, score=score)
                
                # Mark as visited at top level
                max_level = max(0, self.hnsw.max_level - 1)
                self.coordination_service.visited_set.checkAndInsert(
                    node_id=node_id, 
                    level=max_level
                )
                
                # Add to priority queue
                self.coordination_service.priority_queue.insert(
                    node_id=node_id, 
                    level=max_level, 
                    score=score
                )
                
                scored_count += 1
            
            logger.info(f"Primed traversal with {scored_count} top-level nodes")
            
        except Exception as e:
            logger.error(f"Error priming traversal: {e}")
            raise
    
    def traverse(self, n_workers: int, timeout: Optional[float] = None, 
                n_to_score: Optional[int] = None, **kwargs):
        """
        Start the distributed traversal with specified number of workers.
        
        Args:
            n_workers: Number of worker processes to spawn
            timeout: Maximum runtime in seconds
            n_to_score: Maximum number of molecules to score
            **kwargs: Additional arguments passed to workers
        """
        if not self.is_initialized:
            raise RuntimeError("Services not initialized")
        
        if timeout is None and n_to_score is None:
            raise ValueError("Must provide either timeout or n_to_score")
        
        logger.info(f"Starting traversal with {n_workers} workers")
        
        try:
            # Set up termination conditions
            termination_conditions = {}
            if timeout is not None:
                termination_conditions['timeout'] = timeout
            if n_to_score is not None:
                termination_conditions['n_to_score'] = n_to_score
            
            # Start coordination service
            self.coordination_service.start(termination_conditions)
            
            # Create and start workers
            if n_workers == 1:
                # Single worker mode
                worker = DistributedWorker(
                    worker_id=f"{self.namespace}_worker_0",
                    hnsw_service=self.hnsw_service,
                    coordination_service=self.coordination_service,
                    scoring_fn=self.scoring_fn,
                    **kwargs
                )
                
                if not worker.start():
                    raise RuntimeError("Failed to start worker")
                
                self.workers.append(worker)
                
            else:
                # Multi-worker mode using worker pool
                worker_config = {
                    'worker_id_prefix': f"{self.namespace}_worker",
                    'hnsw_service': self.hnsw_service,
                    'coordination_service': self.coordination_service,
                    'scoring_fn': self.scoring_fn,
                    **kwargs
                }
                
                self.worker_pool = create_worker_pool(n_workers, **worker_config)
                
                if not self.worker_pool.start_all():
                    raise RuntimeError("Failed to start worker pool")
            
            self.is_running = True
            
            # Monitor traversal progress
            self._monitor_traversal()
            
        except Exception as e:
            logger.error(f"Error during traversal: {e}")
            self.shutdown()
            raise
    
    def _monitor_traversal(self):
        """Monitor traversal progress and handle termination."""
        logger.info("Monitoring traversal progress...")
        
        while self.is_running:
            try:
                # Check termination conditions
                should_terminate, reason = self.coordination_service.check_termination()
                
                if should_terminate:
                    logger.info(f"Termination condition met: {reason}")
                    break
                
                # Log progress periodically
                stats = self.coordination_service.get_coordination_stats()
                logger.debug(f"Traversal progress: {stats['scored_molecules']} molecules scored, "
                           f"{stats['workers']['active_workers']} active workers")
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error monitoring traversal: {e}")
                break
        
        logger.info("Traversal monitoring complete")
    
    @property
    def scored_set(self):
        """Get the scored set for compatibility with original API."""
        if not self.coordination_service:
            raise RuntimeError("Coordination service not initialized")
        return self.coordination_service.scored_set
    
    @property
    def priority_queue(self):
        """Get the priority queue for compatibility with original API."""
        if not self.coordination_service:
            raise RuntimeError("Coordination service not initialized")
        return self.coordination_service.priority_queue
    
    @property
    def visited_set(self):
        """Get the visited set for compatibility with original API."""
        if not self.coordination_service:
            raise RuntimeError("Coordination service not initialized")
        return self.coordination_service.visited_set
    
    def get_traversal_stats(self) -> Dict[str, Any]:
        """Get comprehensive traversal statistics."""
        stats = {
            'deployment_mode': self.deployment_mode,
            'namespace': self.namespace,
            'is_initialized': self.is_initialized,
            'is_running': self.is_running
        }
        
        if self.coordination_service:
            stats['coordination'] = self.coordination_service.get_coordination_stats()
        
        if self.hnsw_service:
            stats['hnsw_service'] = self.hnsw_service.get_service_info()
        
        if self.worker_pool:
            stats['worker_pool'] = self.worker_pool.get_pool_stats()
        elif self.workers:
            stats['workers'] = [worker.get_worker_stats() for worker in self.workers]
        
        return stats
    
    def shutdown(self, **kwargs):
        """
        Shutdown all services and workers gracefully.
        
        Args:
            **kwargs: Additional shutdown options
        """
        logger.info("Shutting down RAD traverser...")
        
        self.is_running = False
        
        try:
            # Stop workers
            if self.worker_pool:
                self.worker_pool.stop_all()
                self.worker_pool = None
            
            for worker in self.workers:
                worker.stop()
            self.workers.clear()
            
            # Stop coordination service
            if self.coordination_service:
                self.coordination_service.shutdown("Traverser shutdown")
            
            # Stop HNSW service
            if self.hnsw_service:
                self.hnsw_service.shutdown()
            
            # Stop Redis server if local
            if self.redis_server:
                self.redis_server.shutdown(**kwargs)
            
            logger.info("RAD traverser shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Convenience functions for different deployment modes

def create_local_traverser(hnsw, scoring_fn, **kwargs) -> RADTraverser:
    """Create a traverser configured for local deployment."""
    return RADTraverser(
        hnsw=hnsw,
        scoring_fn=scoring_fn,
        deployment_mode="local",
        **kwargs
    )

def create_distributed_traverser(hnsw, scoring_fn, redis_host: str, 
                               redis_port: int = 6379, **kwargs) -> RADTraverser:
    """Create a traverser configured for distributed deployment."""
    return RADTraverser(
        hnsw=hnsw,
        scoring_fn=scoring_fn,
        deployment_mode="distributed",
        redis_host=redis_host,
        redis_port=redis_port,
        **kwargs
    )
