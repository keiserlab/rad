#!/usr/bin/env python3
"""
Test Redis password authentication functionality.

This test verifies that:
1. RedisServer can start with password authentication
2. RADTraverser can connect to password-protected Redis
3. Authentication failures are handled gracefully
4. Backward compatibility is maintained (no password)
"""

import pytest
import redis
import time
import tempfile
import os
from unittest.mock import patch

from rad.redis_server import RedisServer
from rad.traverser import RADTraverser, create_distributed_traverser
from rad.hnsw_service import create_local_hnsw_service

# Create a mock HNSW for testing
class MockHNSW:
    def __init__(self):
        self.max_level = 3
        self.connectivity = 16
        self.dtype = 'float32'
        self.ndim = 256
        self.capacity = 1000
        self.memory_usage = 1024
        self.multi = False

    def __len__(self):
        return 100

    def get_neighbors(self, node_id, level):
        # Return mock neighbors: [neighbor_id, node_key, neighbor_id, node_key, ...]
        return [1, 101, 2, 102, 3, 103]

    def get_top_level_nodes(self):
        # Return mock top level nodes: [node_id, node_key, node_id, node_key, ...]
        return [0, 100, 1, 101, 2, 102]

def test_redis_server_no_password():
    """Test RedisServer works without password (backward compatibility)."""
    # Find an available port
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    
    redis_server = RedisServer(redis_port=port)
    
    try:
        # Should be able to connect without password
        client = redis.StrictRedis(host='localhost', port=port, decode_responses=False)
        assert client.ping() == True
        
        # Should be able to get client from server
        server_client = redis_server.getClient()
        assert server_client.ping() == True
        
    finally:
        redis_server.shutdown()

def test_redis_server_with_password():
    """Test RedisServer works with password authentication."""
    # Find an available port
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    
    test_password = "test_password_123"
    redis_server = RedisServer(redis_port=port, password=test_password)
    
    try:
        # Should NOT be able to connect without password
        client_no_auth = redis.StrictRedis(host='localhost', port=port, decode_responses=False)
        with pytest.raises(redis.AuthenticationError):
            client_no_auth.ping()
        
        # Should be able to connect with correct password
        client_with_auth = redis.StrictRedis(host='localhost', port=port, password=test_password, decode_responses=False)
        assert client_with_auth.ping() == True
        
        # Server client should work (has password configured)
        server_client = redis_server.getClient()
        assert server_client.ping() == True
        
    finally:
        redis_server.shutdown()

def test_redis_server_wrong_password():
    """Test RedisServer rejects wrong password."""
    # Find an available port
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    
    test_password = "correct_password"
    redis_server = RedisServer(redis_port=port, password=test_password)
    
    try:
        # Should NOT be able to connect with wrong password
        client_wrong_auth = redis.StrictRedis(host='localhost', port=port, password="wrong_password", decode_responses=False)
        with pytest.raises(redis.AuthenticationError):
            client_wrong_auth.ping()
            
    finally:
        redis_server.shutdown()

def test_rad_traverser_local_no_password():
    """Test RADTraverser with local Redis (no password) - backward compatibility."""
    # Find an available port
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    
    # Create mock HNSW and service
    mock_hnsw = MockHNSW()
    hnsw_service = create_local_hnsw_service(mock_hnsw)
    
    def mock_scoring_fn(smiles):
        return 0.5  # Mock score
    
    # Create traverser without password
    traverser = RADTraverser(
        hnsw_service=hnsw_service,
        scoring_fn=mock_scoring_fn,
        deployment_mode="local",
        redis_port=port
    )
    
    try:
        # Should initialize successfully
        assert traverser.is_initialized == True
        assert traverser.redis_client.ping() == True
        
    finally:
        traverser.shutdown()

def test_rad_traverser_local_with_password():
    """Test RADTraverser with local Redis and password."""
    # Find an available port
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    
    # Create mock HNSW and service
    mock_hnsw = MockHNSW()
    hnsw_service = create_local_hnsw_service(mock_hnsw)
    
    def mock_scoring_fn(smiles):
        return 0.7  # Mock score
    
    test_password = "traverser_password_456"
    
    # Create traverser with password
    traverser = RADTraverser(
        hnsw_service=hnsw_service,
        scoring_fn=mock_scoring_fn,
        deployment_mode="local",
        redis_port=port,
        redis_password=test_password
    )
    
    try:
        # Should initialize successfully
        assert traverser.is_initialized == True
        assert traverser.redis_client.ping() == True
        
        # Should not be able to connect without password from external client
        external_client = redis.StrictRedis(host='localhost', port=port, decode_responses=False)
        with pytest.raises(redis.AuthenticationError):
            external_client.ping()
            
    finally:
        traverser.shutdown()

def test_rad_traverser_distributed_with_password():
    """Test RADTraverser with distributed Redis and password."""
    # Start a Redis server with password
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    
    test_password = "distributed_password_789"
    redis_server = RedisServer(redis_port=port, password=test_password)
    
    try:
        # Create mock HNSW and service
        mock_hnsw = MockHNSW()
        hnsw_service = create_local_hnsw_service(mock_hnsw)
        
        def mock_scoring_fn(smiles):
            return 0.8  # Mock score
        
        # Create distributed traverser with password
        traverser = RADTraverser(
            hnsw_service=hnsw_service,
            scoring_fn=mock_scoring_fn,
            deployment_mode="distributed",
            redis_host="localhost",
            redis_port=port,
            redis_password=test_password
        )
        
        try:
            # Should initialize successfully
            assert traverser.is_initialized == True
            assert traverser.redis_client.ping() == True
            
        finally:
            traverser.shutdown()
            
    finally:
        redis_server.shutdown()

def test_create_distributed_traverser_with_password():
    """Test create_distributed_traverser factory function with password."""
    # Start a Redis server with password
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    
    test_password = "factory_password_101112"
    redis_server = RedisServer(redis_port=port, password=test_password)
    
    try:
        # Create mock HNSW
        mock_hnsw = MockHNSW()
        
        def mock_scoring_fn(smiles):
            return 0.9  # Mock score
        
        # Use factory function with password
        traverser = create_distributed_traverser(
            hnsw=mock_hnsw,
            scoring_fn=mock_scoring_fn,
            redis_host="localhost",
            redis_port=port,
            redis_password=test_password
        )
        
        try:
            # Should initialize successfully
            assert traverser.is_initialized == True
            assert traverser.redis_client.ping() == True
            
        finally:
            traverser.shutdown()
            
    finally:
        redis_server.shutdown()

def test_authentication_error_handling():
    """Test graceful handling of authentication errors."""
    # Start a Redis server with password
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    
    test_password = "correct_password"
    redis_server = RedisServer(redis_port=port, password=test_password)
    
    try:
        # Create mock HNSW
        mock_hnsw = MockHNSW()
        hnsw_service = create_local_hnsw_service(mock_hnsw)
        
        def mock_scoring_fn(smiles):
            return 0.6  # Mock score
        
        # Try to create traverser with wrong password - should raise exception
        with pytest.raises((redis.AuthenticationError, RuntimeError)):
            traverser = RADTraverser(
                hnsw_service=hnsw_service,
                scoring_fn=mock_scoring_fn,
                deployment_mode="distributed",
                redis_host="localhost",
                redis_port=port,
                redis_password="wrong_password"
            )
            
    finally:
        redis_server.shutdown()

if __name__ == "__main__":
    # Run basic tests
    print("Testing Redis password authentication...")
    
    print("1. Testing RedisServer without password...")
    test_redis_server_no_password()
    print("✓ PASS")
    
    print("2. Testing RedisServer with password...")
    test_redis_server_with_password()
    print("✓ PASS")
    
    print("3. Testing wrong password rejection...")
    test_redis_server_wrong_password()
    print("✓ PASS")
    
    print("4. Testing RADTraverser local mode without password...")
    test_rad_traverser_local_no_password()
    print("✓ PASS")
    
    print("5. Testing RADTraverser local mode with password...")
    test_rad_traverser_local_with_password()
    print("✓ PASS")
    
    print("6. Testing RADTraverser distributed mode with password...")
    test_rad_traverser_distributed_with_password()
    print("✓ PASS")
    
    print("7. Testing factory function with password...")
    test_create_distributed_traverser_with_password()
    print("✓ PASS")
    
    print("8. Testing authentication error handling...")
    test_authentication_error_handling()
    print("✓ PASS")
    
    print("\nAll Redis password authentication tests passed! 🎉")