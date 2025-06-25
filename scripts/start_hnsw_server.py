#!/usr/bin/env python3
"""
HNSW Server Startup Script

Convenience script to start an HNSW HTTP server for distributed deployment.
This script loads an HNSW index and serves it via FastAPI REST API.

SECURITY WARNING: This server is designed for development and research use.
DO NOT deploy directly to production or public internet without security hardening.
See README.md for security considerations before public deployment.

Usage:
    python start_hnsw_server.py --hnsw-path index.usearch --host 0.0.0.0 --port 8000
    python start_hnsw_server.py --test-data --port 8001 --api-key secret123
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to Python path so we can import rad modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_test_hnsw():
    """Create a test HNSW index for demonstration purposes."""
    try:
        import numpy as np
        from usearch.index import Index
    except ImportError:
        raise ImportError("NumPy and usearch are required for test data. Install with: pip install numpy usearch")
    
    print("Creating test HNSW index...")
    
    # Create test data
    n_vectors = 1000
    dim = 64
    
    vectors = np.random.randint(0, 2, size=(n_vectors, dim), dtype=np.uint8)
    packed_vectors = np.packbits(vectors, axis=1)
    
    # Create HNSW index
    hnsw = Index(
        ndim=dim,
        dtype='b1',
        metric='tanimoto', 
        connectivity=4,
        expansion_add=20
    )
    
    keys = np.arange(n_vectors)
    hnsw.add(keys, packed_vectors)
    
    print(f"Created test HNSW index with {n_vectors} vectors")
    return hnsw

def load_hnsw_from_path(path: str):
    """Load HNSW index from file path."""
    try:
        from usearch.index import Index
    except ImportError:
        raise ImportError("usearch is required to load HNSW indices. Install with: pip install usearch")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"HNSW index file not found: {path}")
    
    print(f"Loading HNSW index from {path}...")
    hnsw = Index(path=path, view=True)
    print(f"Loaded HNSW index with {len(hnsw)} vectors")
    return hnsw

def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(
        description="Start HNSW HTTP server for distributed deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with test data
  python start_hnsw_server.py --test-data --port 8000
  
  # Start server with existing HNSW index and database
  python start_hnsw_server.py --hnsw-path /path/to/index.usearch --database-path /path/to/molecules.db --host 0.0.0.0 --port 8000
  
  # Start server with authentication
  python start_hnsw_server.py --test-data --api-key secret123 --port 8001
  
  # Start server in debug mode with database
  python start_hnsw_server.py --test-data --database-path /path/to/molecules.db --debug --port 8002
        """
    )
    
    # HNSW index source
    index_group = parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument(
        '--hnsw-path', 
        type=str,
        help='Path to existing HNSW index file'
    )
    index_group.add_argument(
        '--test-data', 
        action='store_true',
        help='Create test HNSW index with random data'
    )
    
    # Server configuration
    parser.add_argument(
        '--host', 
        type=str, 
        default='127.0.0.1',
        help='Host to bind server to (default: 127.0.0.1 - localhost only for security)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000,
        help='Port to bind server to (default: 8000)'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=1,
        help='Number of worker processes (default: 1)'
    )
    
    # Database configuration
    parser.add_argument(
        '--database-path',
        type=str,
        help='Path to SQLite database file for SMILES lookup'
    )
    
    # Authentication
    parser.add_argument(
        '--api-key', 
        type=str,
        help='Optional API key for authentication'
    )
    
    # Server options
    parser.add_argument(
        '--enable-cors', 
        action='store_true', 
        default=True,
        help='Enable CORS for web access (default: True)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode with detailed logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Load or create HNSW index
        if args.test_data:
            hnsw = create_test_hnsw()
        else:
            hnsw = load_hnsw_from_path(args.hnsw_path)
        
        # Import and start server
        from rad.hnsw_server import run_hnsw_server
        
        logger.info(f"Starting HNSW server on {args.host}:{args.port}")
        if args.api_key:
            logger.info("API key authentication enabled")
        
        # Server configuration
        server_config = {
            'database_path': args.database_path,
            'api_key': args.api_key,
            'enable_cors': args.enable_cors,
            'debug': args.debug
        }
        
        # Start the server
        run_hnsw_server(
            hnsw=hnsw,
            host=args.host,
            port=args.port,
            workers=args.workers,
            **server_config
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()