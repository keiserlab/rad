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
    
    def __init__(self, hnsw, max_queue_size: int = 1000, 
                 response_timeout: float = 30.0, health_check_interval: float = 5.0):
        """
        Initialize the local HNSW service.
        
        Args:
            hnsw: The HNSW index to serve
            max_queue_size: Maximum number of queued requests
            response_timeout: Timeout for waiting for responses (seconds)
            health_check_interval: Interval for health checks (seconds)
        """
        self.hnsw = hnsw
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
    
    def _server_process(self):
        """
        Server process that handles HNSW requests.
        
        This process runs independently and processes requests from the queue.
        Each request is tagged with a unique ID to ensure proper response routing.
        """
        logger.info("HNSW server process started")
        
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
                            response = self.hnsw.get_neighbors(node_id, level)
                        elif request_type == "get_top_level_nodes":
                            response = self.hnsw.get_top_level_nodes()
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
            logger.info("HNSW server process shutting down")
    
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
        **kwargs: Additional arguments for LocalHNSWService
        
    Returns:
        Configured LocalHNSWService instance
    """
    service = LocalHNSWService(hnsw, **kwargs)
    service_registry.register_service("local", service, is_default=True)
    return service


# Future: Remote HNSW service implementation
class RemoteHNSWService(HNSWService):
    """
    Remote HNSW service client for HTTP/API-based access.
    
    This will enable accessing HNSW indices hosted on remote servers,
    cloud APIs, or other network-accessible services.
    
    Note: This is a placeholder for future implementation.
    """
    
    def __init__(self, api_url: str, api_key: Optional[str] = None, 
                 timeout: float = 30.0, **kwargs):
        """
        Initialize remote HNSW service client.
        
        Args:
            api_url: Base URL for the HNSW API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        
        # TODO: Implement HTTP client initialization
        raise NotImplementedError("RemoteHNSWService will be implemented in Phase 2")
    
    def get_neighbors(self, node_id: int, level: int) -> List[int]:
        # TODO: Implement HTTP API call
        raise NotImplementedError("Remote HNSW API not yet implemented")
    
    def get_top_level_nodes(self) -> List[int]:
        # TODO: Implement HTTP API call
        raise NotImplementedError("Remote HNSW API not yet implemented")
    
    def is_healthy(self) -> bool:
        # TODO: Implement health check API call
        raise NotImplementedError("Remote HNSW API not yet implemented")
    
    def shutdown(self) -> None:
        # TODO: Implement connection cleanup
        pass
    
    def get_service_info(self) -> Dict[str, Any]:
        return {
            "service_type": "RemoteHNSWService", 
            "api_url": self.api_url,
            "status": "not_implemented"
        }