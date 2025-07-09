#!/usr/bin/env python3
"""
HNSW HTTP Server

FastAPI-based HTTP server that exposes HNSW operations over REST API.
Enables distributed deployment where HNSW services run on separate machines.

SECURITY WARNING: This server is designed for development and research use.
DO NOT deploy directly to production or public internet without security hardening.

Key Features:
- RESTful API with automatic OpenAPI documentation
- Thread-safe concurrent request handling
- Request correlation with UUIDs for debugging
- Performance metrics and health monitoring
- Basic API key authentication
- Structured logging for distributed systems
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Depends, Security, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    raise ImportError("FastAPI and uvicorn are required for HNSW server. Install with: pip install fastapi uvicorn")

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class HNSWServerMetrics:
    """Metrics collection for HNSW server performance monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_request_time = 0.0
        self.neighbor_queries = 0
        self.top_level_queries = 0
        
    def record_request(self, endpoint: str, duration: float, success: bool = True):
        """Record a completed request with timing and outcome."""
        self.request_count += 1
        self.total_request_time += duration
        
        if endpoint == "neighbors":
            self.neighbor_queries += 1
        elif endpoint == "top_level":
            self.top_level_queries += 1
            
        if not success:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        uptime = time.time() - self.start_time
        avg_request_time = self.total_request_time / max(self.request_count, 1)
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "success_rate": (self.request_count - self.error_count) / max(self.request_count, 1),
            "avg_request_time_seconds": avg_request_time,
            "requests_per_second": self.request_count / max(uptime, 1),
            "neighbor_queries": self.neighbor_queries,
            "top_level_queries": self.top_level_queries
        }


class HNSWServerApp:
    """Main HNSW server application with FastAPI integration and SQLite support."""
    
    def __init__(self, hnsw, database_path: Optional[str] = None, 
                 api_key: Optional[str] = None, enable_cors: bool = True, 
                 debug: bool = False, db_pool_size: int = 10):
        """
        Initialize HNSW server application with SQLite integration.
        
        Args:
            hnsw: HNSW index to serve
            database_path: Path to SQLite database file for SMILES lookup
            api_key: Optional API key for authentication
            enable_cors: Enable CORS middleware for web access
            debug: Enable debug mode with detailed logging
            db_pool_size: Database connection pool size
        """
        self.hnsw = hnsw
        self.database_path = database_path
        self.api_key = api_key
        self.enable_cors = enable_cors
        self.debug = debug
        self.metrics = HNSWServerMetrics()
        
        # Database connection management
        self.db_pool_size = db_pool_size
        self.db_connections = {}
        self.db_lock = threading.Lock()
        self.db_executor = ThreadPoolExecutor(max_workers=db_pool_size)
        
        # Initialize database connection if provided
        if self.database_path:
            self._init_database()
        else:
            logger.warning("No database path provided - SMILES lookup will not be available")
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="HNSW Service API",
            description="HTTP API for HNSW neighbor queries and operations",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            lifespan=self._lifespan
        )
        
        # Setup middleware and authentication first
        self._setup_middleware()
        self._setup_authentication()
        self._setup_routes()
        
        logger.info("HNSWServerApp initialized")
    
    def _init_database(self):
        """Initialize database connection and validate schema."""
        try:
            # Test database connection and schema
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Check if nodes table exists with correct schema
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='nodes'
            """)
            
            if not cursor.fetchone():
                logger.error("Database missing 'nodes' table")
                raise RuntimeError("Database missing required 'nodes' table")
            
            # Check table schema
            cursor.execute("PRAGMA table_info(nodes)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            required_columns = {'node_key': 'INTEGER', 'smi': 'TEXT'}
            for col_name, col_type in required_columns.items():
                if col_name not in columns:
                    logger.error(f"Database missing required column: {col_name}")
                    raise RuntimeError(f"Database missing required column: {col_name}")
            
            # Test query
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = cursor.fetchone()[0]
            
            conn.close()
            
            logger.info(f"Database initialized successfully with {node_count} nodes")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise RuntimeError(f"Database initialization failed: {e}")
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """Get database connection from pool (thread-safe)."""
        thread_id = threading.get_ident()
        
        with self.db_lock:
            if thread_id not in self.db_connections:
                self.db_connections[thread_id] = sqlite3.connect(self.database_path)
                # Enable row factory for easier access
                self.db_connections[thread_id].row_factory = sqlite3.Row
                
        return self.db_connections[thread_id]
    
    async def _get_smiles_batch(self, node_keys: List[int]) -> Dict[int, str]:
        """
        Get SMILES strings for multiple node keys in batch.
        
        Args:
            node_keys: List of node keys to lookup
            
        Returns:
            Dictionary mapping node_key -> smiles
        """
        if not self.database_path:
            logger.warning("No database available for SMILES lookup")
            return {}
        
        if not node_keys:
            return {}
        
        try:
            # Use thread pool executor for database I/O
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.db_executor, 
                self._sync_get_smiles_batch, 
                node_keys
            )
            return result
            
        except Exception as e:
            logger.error(f"Error fetching SMILES for keys {node_keys}: {e}")
            return {}
    
    def _sync_get_smiles_batch(self, node_keys: List[int]) -> Dict[int, str]:
        """Synchronous batch SMILES lookup for use in thread pool."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # Create placeholders for IN query
        placeholders = ','.join(['?' for _ in node_keys])
        query = f"SELECT node_key, smi FROM nodes WHERE node_key IN ({placeholders})"
        
        cursor.execute(query, node_keys)
        results = cursor.fetchall()
        
        # Convert to dictionary
        smiles_map = {row['node_key']: row['smi'] for row in results}
        
        # Log missing SMILES
        missing_keys = set(node_keys) - set(smiles_map.keys())
        if missing_keys:
            logger.warning(f"Missing SMILES for node keys: {missing_keys}")
        
        return smiles_map

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Application lifespan management."""
        logger.info("HNSW server starting...")
        yield
        logger.info("HNSW server shutting down...")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        if self.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        @self.app.middleware("http")
        async def request_logging_middleware(request: Request, call_next):
            """Log all requests with timing and correlation IDs."""
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Add request ID to request state
            request.state.request_id = request_id
            
            logger.info(f"Request {request_id}: {request.method} {request.url}")
            
            try:
                response = await call_next(request)
                duration = time.time() - start_time
                
                logger.info(f"Request {request_id}: {response.status_code} "
                          f"({duration:.3f}s)")
                
                # Record metrics
                endpoint = self._get_endpoint_name(request.url.path)
                self.metrics.record_request(endpoint, duration, 
                                          success=response.status_code < 400)
                
                # Add request ID to response headers
                response.headers["X-Request-ID"] = request_id
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Request {request_id}: Error - {e} ({duration:.3f}s)")
                
                # Record error metrics
                endpoint = self._get_endpoint_name(request.url.path)
                self.metrics.record_request(endpoint, duration, success=False)
                
                raise
    
    def _get_endpoint_name(self, path: str) -> str:
        """Extract endpoint name from URL path for metrics."""
        if "/neighbors/" in path:
            return "neighbors"
        elif "/top-level-nodes" in path:
            return "top_level"
        elif "/health" in path:
            return "health"
        elif "/info" in path:
            return "info"
        elif "/ping" in path:
            return "ping"
        else:
            return "unknown"
    
    def _setup_authentication(self):
        """Setup API key authentication if enabled."""
        if self.api_key:
            security = HTTPBearer()
            
            def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
                if credentials.credentials != self.api_key:
                    raise HTTPException(status_code=401, detail="Invalid API key")
                return credentials.credentials
            
            self.auth_dependency = verify_api_key
        else:
            self.auth_dependency = lambda: None
    
    def _setup_routes(self):
        """Setup FastAPI routes for HNSW operations."""
        
        @self.app.get("/")
        async def serve_homepage():
            """Serve the homepage HTML file."""
            import os
            # Look for index.html in current directory
            index_path = os.path.join(os.getcwd(), "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path, media_type="text/html")
            else:
                return {"message": "RAD HNSW Service", "status": "running", "docs": "/docs"}
        
        
        @self.app.get("/{filename}")
        async def serve_static_files(filename: str):
            """Serve static files like images from current directory."""
            import os
            # Only serve common static file types for security
            allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.css', '.js', '.ico'}
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in allowed_extensions:
                file_path = os.path.join(os.getcwd(), filename)
                if os.path.exists(file_path):
                    return FileResponse(file_path)
            
            # If not a static file or doesn't exist, return 404
            raise HTTPException(status_code=404, detail="File not found")
        
        @self.app.get("/ping")
        async def ping():
            return {"pong": True}

        @self.app.get("/neighbors/{node_id}/{level}")
        async def get_neighbors(
            node_id: int, 
            level: int,
            request: Request,
            api_key: str = Depends(self.auth_dependency)
        ):
            """
            Get neighbors for a node at a specific HNSW level with SMILES.
            
            Returns alternating list of [neighbor_id, smiles, neighbor_id, smiles, ...].
            """
            try:
                # Validate inputs
                if node_id < 0:
                    raise HTTPException(status_code=400, detail="node_id must be non-negative")
                if level < 0:
                    raise HTTPException(status_code=400, detail="level must be non-negative")
                
                # Query HNSW for neighbors (returns [neighbor_id, neighbor_key, ...])
                hnsw_neighbors = [int(x) for x in self.hnsw.get_neighbors(node_id, level)]
                
                # Extract node_keys for SMILES lookup
                node_keys = [hnsw_neighbors[i+1] for i in range(0, len(hnsw_neighbors), 2)]
                neighbor_ids = [hnsw_neighbors[i] for i in range(0, len(hnsw_neighbors), 2)]
                
                # Get SMILES for all node_keys in batch
                smiles_map = await self._get_smiles_batch(node_keys)
                
                # Build response with [neighbor_id, smiles, ...] format
                neighbors_with_smiles = []
                for neighbor_id, node_key in zip(neighbor_ids, node_keys):
                    smiles = smiles_map.get(node_key, "")  # Empty string if SMILES not found
                    neighbors_with_smiles.extend([neighbor_id, smiles])
                
                logger.debug(f"Request {request.state.request_id}: "
                           f"Found {len(neighbor_ids)} neighbors for node {node_id} at level {level}")
                
                return {
                    "node_id": node_id,
                    "level": level,
                    "neighbors": neighbors_with_smiles,
                    "neighbor_count": len(neighbor_ids),
                    "request_id": request.state.request_id
                }
                
            except Exception as e:
                logger.error(f"Error getting neighbors for node {node_id}, level {level}: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/top-level-nodes")
        async def get_top_level_nodes(
            request: Request,
            api_key: str = Depends(self.auth_dependency)
        ):
            """
            Get top-level nodes for traversal priming with SMILES.
            
            Returns alternating list of [node_id, smiles, node_id, smiles, ...].
            """
            try:
                # Get top level nodes from HNSW (returns [node_id, node_key, ...])
                hnsw_top_nodes = [int(x) for x in self.hnsw.get_top_level_nodes()]
                
                # Extract node_keys for SMILES lookup
                node_keys = [hnsw_top_nodes[i+1] for i in range(0, len(hnsw_top_nodes), 2)]
                node_ids = [hnsw_top_nodes[i] for i in range(0, len(hnsw_top_nodes), 2)]
                
                # Get SMILES for all node_keys in batch
                smiles_map = await self._get_smiles_batch(node_keys)
                
                # Build response with [node_id, smiles, ...] format
                top_nodes_with_smiles = []
                for node_id, node_key in zip(node_ids, node_keys):
                    smiles = smiles_map.get(node_key, "")  # Empty string if SMILES not found
                    top_nodes_with_smiles.extend([node_id, smiles])
                
                logger.debug(f"Request {request.state.request_id}: "
                           f"Found {len(node_ids)} top-level nodes")
                
                return {
                    "top_nodes": top_nodes_with_smiles,
                    "node_count": len(node_ids),
                    "request_id": request.state.request_id
                }
                
            except Exception as e:
                logger.error(f"Error getting top-level nodes: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get("/health")
        async def health_check(request: Request):
            """
            Health check endpoint for service monitoring.
            
            Returns service health status and basic metrics.
            """
            try:
                # Basic health checks
                hnsw_size = int(len(self.hnsw))
                hnsw_max_level = int(self.hnsw.max_level)
                
                health_status = {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "hnsw_size": hnsw_size,
                    "hnsw_max_level": hnsw_max_level,
                    "uptime_seconds": time.time() - self.metrics.start_time,
                    "request_id": request.state.request_id
                }
                
                return health_status
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy", 
                        "error": str(e),
                        "timestamp": time.time(),
                        "request_id": getattr(request.state, 'request_id', 'unknown')
                    }
                )
        
        @self.app.get("/info")
        async def get_service_info(
            request: Request,
            api_key: str = Depends(self.auth_dependency)
        ):
            """
            Get detailed service information and performance metrics.
            """
            try:
                hnsw_info = {
                    "max_level": int(self.hnsw.max_level),
                    "size": int(len(self.hnsw)),
                    "connectivity": int(self.hnsw.connectivity),
                    "dtype": str(self.hnsw.dtype),
                    "ndim": int(self.hnsw.ndim),
                    "capacity": int(self.hnsw.capacity),
                    "memory_usage": int(self.hnsw.memory_usage),
                    "multi": bool(self.hnsw.multi)
                }
                
                service_info = {
                    "service_type": "RemoteHNSWService",
                    "version": "1.0.0",
                    "hnsw_info": hnsw_info,
                    "performance_metrics": self.metrics.get_stats(),
                    "authentication_enabled": self.api_key is not None,
                    "cors_enabled": self.enable_cors,
                    "debug_mode": self.debug,
                    "request_id": request.state.request_id
                }
                
                return service_info
                
            except Exception as e:
                logger.error(f"Error getting service info: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def create_hnsw_server(hnsw, **kwargs) -> HNSWServerApp:
    """
    Create an HNSW server application.
    
    Args:
        hnsw: HNSW index to serve
        **kwargs: Additional configuration options
        
    Returns:
        Configured HNSWServerApp instance
    """
    return HNSWServerApp(hnsw, **kwargs)


def run_hnsw_server(hnsw, host: str = "0.0.0.0", port: int = 8000, 
                   workers: int = 1, **kwargs):
    """
    Run HNSW server with uvicorn.
    
    Args:
        hnsw: HNSW index to serve
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes (1 for development)
        **kwargs: Additional server configuration
    """
    server_app = create_hnsw_server(hnsw, **kwargs)
    
    logger.info(f"Starting HNSW server on {host}:{port}")
    
    uvicorn.run(
        server_app.app,
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    # Example usage
    import sys
    print("This module provides HNSWServer class and utilities.")
    print("Use run_hnsw_server(hnsw, host='0.0.0.0', port=8000) to start a server.")
    sys.exit(0)