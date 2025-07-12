#!/usr/bin/env python3
"""
HNSW Service Abstraction Layer

This module provides a service-oriented interface for HNSW operations that supports
both local process-based servers and future remote API-based services. This design
enables flexible deployment across single machines, HPC clusters, and cloud environments.

Service Types:
- LocalHNSWService: Process-based server for single-machine or HPC head node deployment
- RemoteHNSWService: HTTP/API-based client for cloud and distributed deployment (future)
"""

import multiprocessing
import uuid
import time
import threading
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

class HNSWService(ABC):
    """
    Abstract interface for HNSW services.
    
    This interface abstracts HNSW operations to support multiple deployment modes:
    - Local: Process-based server on same machine
    - Remote: HTTP/API calls to external HNSW service
    - Hybrid: Local cache with remote fallback
    """
    
    @abstractmethod
    def get_neighbors(self, node_id: int, level: int) -> List[int]:
        """
        Get neighbors for a node at a specific level.
        
        Args:
            node_id: The node to get neighbors for
            level: The HNSW level to query
            
        Returns:
            List of neighbor node IDs and keys (alternating: [neighbor_id, neighbor_key, ...])
        """
        pass
    
    @abstractmethod
    def get_top_level_nodes(self) -> List[int]:
        """
        Get nodes from the top level of the HNSW graph.
        
        Returns:
            List of top level node IDs and keys (alternating: [node_id, node_key, ...])
        """
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the service is healthy and responsive."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        pass
    
    @abstractmethod
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service (type, status, performance metrics, etc.)."""
        pass
    
    @abstractmethod
    def get_hnsw_info(self) -> Dict[str, Any]:
        """Get HNSW metadata (max_level, size, connectivity, etc.)."""
        pass


class LocalHNSWService(HNSWService):
    """
    Local HNSW service that runs in a separate process.
    
    This implementation provides thread-safe access to an HNSW index via a dedicated
    server process. Multiple workers can make concurrent requests without interference.
    
    Features:
    - Request/response tracking with UUIDs to prevent cross-talk
    - Concurrent request handling with proper response routing
    - Health monitoring and performance metrics
    - Graceful shutdown with cleanup
    """
    
    def __init__(self, hnsw, database_path: Optional[str] = None, 
                 max_queue_size: int = 1000, response_timeout: float = 30.0, 
                 health_check_interval: float = 5.0):
        """
        Initialize the local HNSW service.
        
        Args:
            hnsw: The HNSW index to serve
            database_path: Path to SQLite database file for SMILES lookup
            max_queue_size: Maximum number of queued requests
            response_timeout: Timeout for waiting for responses (seconds)
            health_check_interval: Interval for health checks (seconds)
        """
        self.hnsw = hnsw
        self.database_path = database_path
        self.max_queue_size = max_queue_size
        self.response_timeout = response_timeout
        self.health_check_interval = health_check_interval
        
        # Inter-process communication
        self.request_queue = multiprocessing.Queue(maxsize=max_queue_size)
        self.response_queue = multiprocessing.Queue(maxsize=max_queue_size * 2)
        
        # Service state
        self.is_running = False
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # Response tracking for concurrent requests
        self.pending_requests = {}
        self.response_lock = threading.Lock()
        
        # Start the server process
        self.process = multiprocessing.Process(
            target=self._server_process, 
            daemon=True,
            name="HNSWService"
        )
        self.process.start()
        self.is_running = True
        
        # Start response handler thread
        self.response_thread = threading.Thread(
            target=self._response_handler,
            daemon=True,
            name="HNSWResponseHandler"
        )
        self.response_thread.start()
        
        logger.info(f"LocalHNSWService started with PID {self.process.pid}")
    
    def _init_database_in_process(self):
        """Initialize database connection in the server process."""
        if not self.database_path:
            logger.info("No database path provided - SMILES lookup disabled")
            return None
        
        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row
            
            # Test database
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = cursor.fetchone()[0]
            logger.info(f"Database connected with {node_count} nodes")
            
            return conn
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return None
    
    def _get_smiles_batch(self, db_conn, node_keys: List[int]) -> Dict[int, str]:
        """Get SMILES for multiple node keys from database."""
        if not db_conn or not node_keys:
            return {}
        
        try:
            cursor = db_conn.cursor()
            placeholders = ','.join(['?' for _ in node_keys])
            query = f"SELECT node_key, smi FROM nodes WHERE node_key IN ({placeholders})"
            
            cursor.execute(query, node_keys)
            results = cursor.fetchall()
            
            smiles_map = {row['node_key']: row['smi'] for row in results}
            
            # Log missing SMILES
            missing_keys = set(node_keys) - set(smiles_map.keys())
            if missing_keys:
                logger.warning(f"Missing SMILES for node keys: {missing_keys}")
            
            return smiles_map
            
        except Exception as e:
            logger.error(f"Error fetching SMILES: {e}")
            return {}
    
    def _server_process(self):
        """
        Server process that handles HNSW requests with SMILES lookup.
        
        This process runs independently and processes requests from the queue.
        Each request is tagged with a unique ID to ensure proper response routing.
        """
        logger.info("HNSW server process started")
        
        # Initialize database connection in this process
        db_conn = self._init_database_in_process()
        
        try:
            while True:
                try:
                    request = self.request_queue.get(timeout=1.0)
                    
                    if request == "STOP":
                        logger.info("HNSW server received stop signal")
                        break
                    
                    request_id, request_type, args = request
                    
                    try:
                        if request_type == "get_neighbors":
                            node_id, level = args
                            # Get HNSW neighbors (returns [neighbor_id, neighbor_key, ...])
                            hnsw_neighbors = [int(x) for x in self.hnsw.get_neighbors(node_id, level)]
                            
                            # Transform to [neighbor_id, smiles, ...] format
                            response = self._transform_to_smiles_format(db_conn, hnsw_neighbors)
                            
                        elif request_type == "get_top_level_nodes":
                            # Get HNSW top nodes (returns [node_id, node_key, ...])
                            hnsw_top_nodes = [int(x) for x in self.hnsw.get_top_level_nodes()]
                            
                            # Transform to [node_id, smiles, ...] format
                            response = self._transform_to_smiles_format(db_conn, hnsw_top_nodes)
                            
                        elif request_type == "health_check":
                            response = {"status": "healthy", "timestamp": time.time()}
                        else:
                            response = {"error": f"Unknown request type: {request_type}"}
                        
                        self.response_queue.put((request_id, "success", response))
                        
                    except Exception as e:
                        logger.error(f"Error processing request {request_id}: {e}")
                        self.response_queue.put((request_id, "error", str(e)))
                
                except:
                    # Timeout waiting for request - continue loop
                    continue
                    
        except Exception as e:
            logger.error(f"HNSW server process error: {e}")
        finally:
            if db_conn:
                db_conn.close()
            logger.info("HNSW server process shutting down")
    
    def _transform_to_smiles_format(self, db_conn, hnsw_data: List[int]) -> List:
        """
        Transform HNSW data from [node_id, node_key, ...] to [node_id, smiles, ...] format.
        
        Args:
            db_conn: Database connection
            hnsw_data: List in format [node_id, node_key, node_id, node_key, ...]
            
        Returns:
            List in format [node_id, smiles, node_id, smiles, ...]
        """
        if not hnsw_data:
            return []
        
        # Extract node_keys for SMILES lookup
        node_keys = [hnsw_data[i+1] for i in range(0, len(hnsw_data), 2)]
        node_ids = [hnsw_data[i] for i in range(0, len(hnsw_data), 2)]
        
        # Get SMILES for all node_keys
        smiles_map = self._get_smiles_batch(db_conn, node_keys)
        
        # Build result with [node_id, smiles, ...] format
        result = []
        for node_id, node_key in zip(node_ids, node_keys):
            smiles = smiles_map.get(node_key, "")  # Empty string if SMILES not found
            result.extend([node_id, smiles])
        
        return result
    
    def _response_handler(self):
        """
        Background thread that handles responses from the server process.
        
        This thread continuously processes responses and routes them to the
        correct waiting request based on the request ID.
        """
        while self.is_running:
            try:
                response = self.response_queue.get(timeout=1.0)
                request_id, status, data = response
                
                with self.response_lock:
                    if request_id in self.pending_requests:
                        # Set the response for the waiting request
                        event, result_container = self.pending_requests[request_id]
                        result_container['status'] = status
                        result_container['data'] = data
                        event.set()
                    else:
                        logger.warning(f"Received response for unknown request: {request_id}")
                        
            except:
                # Timeout waiting for response - continue loop
                continue
    
    def _make_request(self, request_type: str, args: Tuple = ()) -> Any:
        """
        Make a request to the HNSW server and wait for response.
        
        Args:
            request_type: Type of request ("get_neighbors", "get_top_level_nodes", etc.)
            args: Arguments for the request
            
        Returns:
            Response data from the server
            
        Raises:
            RuntimeError: If service is not running or request fails
            TimeoutError: If request times out
        """
        if not self.is_running:
            raise RuntimeError("HNSW service is not running")
        
        request_id = str(uuid.uuid4())
        result_container = {}
        event = threading.Event()
        
        # Register the request for response tracking
        with self.response_lock:
            self.pending_requests[request_id] = (event, result_container)
        
        try:
            # Send the request
            self.request_queue.put((request_id, request_type, args), timeout=5.0)
            self.request_count += 1
            
            # Wait for response
            if event.wait(timeout=self.response_timeout):
                status = result_container.get('status')
                data = result_container.get('data')
                
                if status == "success":
                    return data
                else:
                    self.error_count += 1
                    raise RuntimeError(f"HNSW request failed: {data}")
            else:
                self.error_count += 1
                raise TimeoutError(f"HNSW request timed out after {self.response_timeout} seconds")
                
        finally:
            # Clean up the pending request
            with self.response_lock:
                self.pending_requests.pop(request_id, None)
    
    def get_neighbors(self, node_id: int, level: int) -> List[int]:
        """Get neighbors for a node at a specific level."""
        return self._make_request("get_neighbors", (node_id, level))
    
    def get_top_level_nodes(self) -> List[int]:
        """Get nodes from the top level of the HNSW graph."""
        return self._make_request("get_top_level_nodes")
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy and responsive."""
        if not self.is_running or not self.process.is_alive():
            return False
        
        try:
            health_data = self._make_request("health_check")
            return health_data.get("status") == "healthy"
        except:
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service."""
        uptime = time.time() - self.start_time
        
        return {
            "service_type": "LocalHNSWService",
            "status": "running" if self.is_running else "stopped",
            "process_id": self.process.pid if self.process else None,
            "process_alive": self.process.is_alive() if self.process else False,
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "pending_requests": len(self.pending_requests),
            "queue_sizes": {
                "request_queue": self.request_queue.qsize(),
                "response_queue": self.response_queue.qsize()
            }
        }
    
    def get_hnsw_info(self) -> Dict[str, Any]:
        """Get HNSW metadata using usearch index properties."""
        try:
            return {
                "max_level": self.hnsw.max_level,
                "size": len(self.hnsw),
                "connectivity": self.hnsw.connectivity,
                "dtype": str(self.hnsw.dtype),
                "ndim": self.hnsw.ndim,
                "capacity": self.hnsw.capacity,
                "memory_usage": self.hnsw.memory_usage,
                "multi": self.hnsw.multi
            }
        except Exception as e:
            logger.error(f"Error getting HNSW info: {e}")
            return {
                "max_level": 0,
                "size": -1,
                "connectivity": -1,
                "dtype": "unknown", 
                "ndim": -1,
                "capacity": -1,
                "memory_usage": -1,
                "multi": False,
                "error": str(e)
            }
    
    def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        if not self.is_running:
            return
        
        logger.info("Shutting down LocalHNSWService...")
        self.is_running = False
        
        # Stop the server process
        try:
            self.request_queue.put("STOP", timeout=2.0)
            self.process.join(timeout=5.0)
            
            if self.process.is_alive():
                logger.warning("HNSW server process did not stop gracefully, terminating...")
                self.process.terminate()
                self.process.join(timeout=2.0)
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        # Wait for response thread to finish
        if self.response_thread.is_alive():
            self.response_thread.join(timeout=2.0)
        
        logger.info("LocalHNSWService shutdown complete")


class RemoteHNSWService(HNSWService):
    """
    Remote HNSW service that connects to HTTP-based HNSW servers.
    
    This implementation enables distributed deployment where HNSW operations
    are performed on remote servers via HTTP REST API calls.
    
    Features:
    - HTTP client with connection pooling for performance
    - Request retry logic with exponential backoff
    - Circuit breaker pattern for fault tolerance
    - Request correlation UUIDs for debugging
    - Configurable timeouts and error handling
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 timeout: float = 30.0, max_retries: int = 3,
                 pool_connections: int = 10, pool_maxsize: int = 20,
                 backoff_factor: float = 0.5):
        """
        Initialize remote HNSW service client.
        
        Args:
            base_url: Base URL of the HNSW server (e.g., "http://localhost:8000")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            pool_connections: Number of connection pools to cache
            pool_maxsize: Maximum number of connections in each pool
            backoff_factor: Exponential backoff factor for retries
        """
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from requests.packages.urllib3.util.retry import Retry
        except ImportError:
            raise ImportError("requests library is required for RemoteHNSWService. Install with: pip install requests")
        
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Setup HTTP session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        
        # Setup connection pooling
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Setup authentication headers
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })
        
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RAD-RemoteHNSWService/1.0.0'
        })
        
        # Service state
        self.is_connected = False
        self.last_health_check = 0
        self.health_check_interval = 60.0  # seconds
        self.request_count = 0
        self.error_count = 0
        
        # Test initial connection
        self._verify_connection()
        
        logger.info(f"RemoteHNSWService initialized for {base_url}")
    
    def _verify_connection(self):
        """Verify connection to remote HNSW service."""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            health_data = response.json()
            if health_data.get('status') == 'healthy':
                self.is_connected = True
                self.last_health_check = time.time()
                logger.info(f"Successfully connected to HNSW service at {self.base_url}")
            else:
                raise RuntimeError(f"HNSW service reported unhealthy status: {health_data}")
                
        except Exception as e:
            logger.error(f"Failed to connect to HNSW service at {self.base_url}: {e}")
            raise RuntimeError(f"Cannot connect to HNSW service: {e}")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """
        Make HTTP request with error handling and metrics.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests
            
        Returns:
            Response JSON data
            
        Raises:
            RuntimeError: On request failure
        """
        url = f"{self.base_url}{endpoint}"
        request_id = str(uuid.uuid4())
        
        # Add request correlation header
        headers = kwargs.get('headers', {})
        headers['X-Correlation-ID'] = request_id
        kwargs['headers'] = headers
        kwargs['timeout'] = kwargs.get('timeout', self.timeout)
        
        try:
            self.request_count += 1
            logger.debug(f"Request {request_id}: {method} {url}")
            
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Request {request_id}: Success ({response.status_code})")
            
            return data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Request {request_id}: Failed - {e}")
            
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json().get('detail', str(e))
                except:
                    error_detail = str(e)
                raise RuntimeError(f"HNSW service error: {error_detail}")
            else:
                raise RuntimeError(f"Network error connecting to HNSW service: {e}")
    
    def get_neighbors(self, node_id: int, level: int) -> List[int]:
        """
        Get neighbors for a node at a specific level via HTTP API.
        
        Args:
            node_id: The node to get neighbors for
            level: The HNSW level to query
            
        Returns:
            List of neighbor node IDs and SMILES (alternating: [neighbor_id, smiles, ...])
        """
        try:
            data = self._make_request('GET', f'/neighbors/{node_id}/{level}')
            neighbors = data.get('neighbors', [])
            
            logger.debug(f"Retrieved {len(neighbors)//2} neighbors with SMILES for node {node_id} at level {level}")
            return neighbors
            
        except Exception as e:
            logger.error(f"Error getting neighbors for node {node_id}, level {level}: {e}")
            raise
    
    def get_top_level_nodes(self) -> List[int]:
        """
        Get top-level nodes for traversal priming via HTTP API.
        
        Returns:
            List of top-level node IDs and SMILES (alternating: [node_id, smiles, ...])
        """
        try:
            data = self._make_request('GET', '/top-level-nodes')
            top_nodes = data.get('top_nodes', [])
            
            logger.debug(f"Retrieved {len(top_nodes)//2} top-level nodes with SMILES")
            return top_nodes
            
        except Exception as e:
            logger.error(f"Error getting top-level nodes: {e}")
            raise
    
    def is_healthy(self) -> bool:
        """
        Check if the remote service is healthy and responsive.
        
        Returns:
            True if service is healthy, False otherwise
        """
        # Check if we need to refresh health status
        current_time = time.time()
        if current_time - self.last_health_check > self.health_check_interval:
            try:
                data = self._make_request('GET', '/health')
                self.is_connected = data.get('status') == 'healthy'
                self.last_health_check = current_time
                
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
                self.is_connected = False
        
        return self.is_connected
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the remote service.
        
        Returns:
            Dictionary with service information and metrics
        """
        try:
            # Get remote service info
            remote_info = self._make_request('GET', '/info')
            
            # Add client-side information
            client_info = {
                "service_type": "RemoteHNSWService",
                "base_url": self.base_url,
                "is_connected": self.is_connected,
                "client_request_count": self.request_count,
                "client_error_count": self.error_count,
                "client_success_rate": (self.request_count - self.error_count) / max(self.request_count, 1),
                "authentication_enabled": self.api_key is not None,
                "last_health_check": self.last_health_check
            }
            
            return {
                "client_info": client_info,
                "remote_service_info": remote_info
            }
            
        except Exception as e:
            logger.error(f"Error getting service info: {e}")
            # Return minimal info if remote call fails
            return {
                "service_type": "RemoteHNSWService",
                "base_url": self.base_url,
                "is_connected": self.is_connected,
                "error": str(e)
            }
    
    def get_hnsw_info(self) -> Dict[str, Any]:
        """Get HNSW metadata from remote service."""
        try:
            # Get remote service info which includes HNSW metadata
            remote_info = self._make_request('GET', '/info')
            hnsw_info = remote_info.get('hnsw_info', {})
            
            return {
                "max_level": hnsw_info.get('max_level', 0),
                "size": hnsw_info.get('size', -1),
                "connectivity": hnsw_info.get('connectivity', -1),
                "dtype": hnsw_info.get('dtype', 'unknown'),
                "ndim": hnsw_info.get('ndim', -1),
                "capacity": hnsw_info.get('capacity', -1),
                "memory_usage": hnsw_info.get('memory_usage', -1),
                "multi": hnsw_info.get('multi', False),
                "remote": True
            }
            
        except Exception as e:
            logger.error(f"Error getting remote HNSW info: {e}")
            return {
                "max_level": 0,
                "size": -1,
                "connectivity": -1,
                "dtype": "unknown", 
                "ndim": -1,
                "capacity": -1,
                "memory_usage": -1,
                "multi": False,
                "remote": True,
                "error": str(e)
            }
    
    def shutdown(self) -> None:
        """
        Shutdown the remote service client gracefully.
        """
        logger.info("Shutting down RemoteHNSWService...")
        
        try:
            self.session.close()
            self.is_connected = False
            logger.info("RemoteHNSWService shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during RemoteHNSWService shutdown: {e}")


class ServiceRegistry:
    """
    Registry for HNSW service instances and configuration.
    
    This class manages service discovery and configuration for different
    deployment environments (local, HPC, cloud).
    """
    
    def __init__(self):
        self.services = {}
        self.default_service = None
    
    def register_service(self, name: str, service: HNSWService, 
                        is_default: bool = False) -> None:
        """Register an HNSW service."""
        self.services[name] = service
        if is_default or self.default_service is None:
            self.default_service = name
        
        logger.info(f"Registered HNSW service: {name}")
    
    def get_service(self, name: Optional[str] = None) -> HNSWService:
        """Get an HNSW service by name, or the default service."""
        service_name = name or self.default_service
        
        if service_name not in self.services:
            raise ValueError(f"HNSW service not found: {service_name}")
        
        return self.services[service_name]
    
    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """List all registered services with their info."""
        return {
            name: service.get_service_info() 
            for name, service in self.services.items()
        }
    
    def shutdown_all(self) -> None:
        """Shutdown all registered services."""
        for name, service in self.services.items():
            try:
                service.shutdown()
                logger.info(f"Shutdown service: {name}")
            except Exception as e:
                logger.error(f"Error shutting down service {name}: {e}")
        
        self.services.clear()
        self.default_service = None


# Global service registry instance
service_registry = ServiceRegistry()


def create_local_hnsw_service(hnsw, **kwargs) -> LocalHNSWService:
    """
    Convenience function to create and register a local HNSW service.
    
    Args:
        hnsw: The HNSW index to serve
        database_path: Path to SQLite database for SMILES lookup (optional)
        **kwargs: Additional arguments for LocalHNSWService
        
    Returns:
        Configured LocalHNSWService instance
    """
    service = LocalHNSWService(hnsw, **kwargs)
    service_registry.register_service("local", service, is_default=True)
    return service


def create_remote_hnsw_service(base_url: str, service_name: Optional[str] = None, 
                             is_default: bool = False, **kwargs) -> RemoteHNSWService:
    """
    Convenience function to create and register a remote HNSW service.
    
    Args:
        base_url: Base URL of the remote HNSW server
        service_name: Name to register service under (auto-generated if None)
        is_default: Whether to set as default service
        **kwargs: Additional arguments for RemoteHNSWService
        
    Returns:
        Configured RemoteHNSWService instance
    """
    service = RemoteHNSWService(base_url, **kwargs)
    
    if service_name is None:
        # Generate service name from URL
        import urllib.parse
        parsed = urllib.parse.urlparse(base_url)
        service_name = f"remote_{parsed.hostname}_{parsed.port or 80}"
    
    service_registry.register_service(service_name, service, is_default=is_default)
    return service