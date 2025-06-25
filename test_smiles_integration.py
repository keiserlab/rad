#!/usr/bin/env python3
"""
Tests for SMILES Integration in HNSW Server

Tests the SQLite integration and new API response format to ensure
SMILES are correctly retrieved and returned in [node_id, smiles, ...] format.
"""

import os
import sqlite3
import tempfile
import threading
import time
import numpy as np
from usearch.index import Index
import pytest

def create_test_database(db_path: str, n_nodes: int = 100):
    """Create a test SQLite database with node_id -> SMILES mapping."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create nodes table
    cursor.execute("""
        CREATE TABLE nodes (
            node_id INTEGER PRIMARY KEY,
            smi TEXT NOT NULL
        )
    """)
    
    # Insert test SMILES data
    test_smiles = [
        "CCO",  # ethanol
        "CCC",  # propane
        "CC(C)C",  # isobutane
        "c1ccccc1",  # benzene
        "CC(=O)O",  # acetic acid
    ]
    
    for i in range(n_nodes):
        # Cycle through test SMILES for variety
        smiles = test_smiles[i % len(test_smiles)]
        # Add some variation
        if i > 0:
            smiles = f"{smiles}.{i}"  # Simple variation for testing
        cursor.execute("INSERT INTO nodes (node_id, smi) VALUES (?, ?)", (i, smiles))
    
    # Create index for performance
    cursor.execute("CREATE INDEX idx_nodes_node_id ON nodes(node_id)")
    
    conn.commit()
    conn.close()
    print(f"Created test database with {n_nodes} nodes at {db_path}")

def create_test_hnsw(n_vectors: int = 100):
    """Create a test HNSW index for testing."""
    dim = 64
    
    vectors = np.random.randint(0, 2, size=(n_vectors, dim), dtype=np.uint8)
    packed_vectors = np.packbits(vectors, axis=1)
    
    hnsw = Index(
        ndim=dim,
        dtype='b1',
        metric='tanimoto', 
        connectivity=4,
        expansion_add=20
    )
    
    keys = np.arange(n_vectors)
    hnsw.add(keys, packed_vectors)
    
    return hnsw

def test_database_creation():
    """Test creation of test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        create_test_database(db_path, 50)
        
        # Verify database contents
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM nodes")
        count = cursor.fetchone()[0]
        assert count == 50
        
        cursor.execute("SELECT node_id, smi FROM nodes WHERE node_id < 5")
        results = cursor.fetchall()
        assert len(results) == 5
        
        # Check first few SMILES
        expected_smiles = ["CCO", "CCC.1", "CC(C)C.2", "c1ccccc1.3", "CC(=O)O.4"]
        for (node_id, smi), expected in zip(results, expected_smiles):
            assert smi == expected, f"Expected {expected}, got {smi} for node_id {node_id}"
        
        conn.close()
        print("✓ Database creation test passed")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_hnsw_server_with_database():
    """Test HNSW server with SQLite database integration."""
    try:
        from rad.hnsw_server import HNSWServerApp
    except ImportError:
        print("❌ Cannot import HNSWServerApp - skipping test")
        return
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        create_test_database(db_path, 20)
        hnsw = create_test_hnsw(20)
        
        # Create server with database
        server_app = HNSWServerApp(
            hnsw=hnsw,
            database_path=db_path,
            debug=True
        )
        
        print("✓ HNSW server created with database integration")
        
        # Test database connection
        assert server_app.database_path == db_path
        assert hasattr(server_app, 'db_connections')
        
        # Test SMILES batch lookup
        import asyncio
        
        async def test_smiles_lookup():
            node_keys = [0, 1, 2, 99]  # Include one that doesn't exist
            smiles_map = await server_app._get_smiles_batch(node_keys)
            
            # Check results
            assert 0 in smiles_map
            assert 1 in smiles_map
            assert 2 in smiles_map
            assert 99 not in smiles_map  # Doesn't exist
            
            assert smiles_map[0] == "CCO"
            assert smiles_map[1] == "CCC.1"
            assert smiles_map[2] == "CC(C)C.2"
            
            return smiles_map
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        smiles_map = loop.run_until_complete(test_smiles_lookup())
        loop.close()
        
        print(f"✓ SMILES lookup test passed: {smiles_map}")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_api_response_format():
    """Test that API endpoints return the new [node_id, smiles, ...] format."""
    try:
        from rad.hnsw_server import HNSWServerApp
        from fastapi.testclient import TestClient
    except ImportError:
        print("❌ Cannot import required modules - skipping API test")
        return
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        create_test_database(db_path, 20)
        hnsw = create_test_hnsw(20)
        
        # Create server
        server_app = HNSWServerApp(
            hnsw=hnsw,
            database_path=db_path,
            debug=True
        )
        
        # Create test client
        client = TestClient(server_app.app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        print("✓ Health endpoint works")
        
        # Test top-level-nodes endpoint
        response = client.get("/top-level-nodes")
        assert response.status_code == 200
        
        data = response.json()
        assert "top_nodes" in data
        assert "node_count" in data
        
        top_nodes = data["top_nodes"]
        node_count = data["node_count"]
        
        # Check format: should be [node_id, smiles, node_id, smiles, ...]
        assert len(top_nodes) == node_count * 2
        assert isinstance(top_nodes[0], int)  # node_id should be int
        assert isinstance(top_nodes[1], str)  # smiles should be string
        
        print(f"✓ Top-level-nodes format correct: {node_count} nodes")
        print(f"  Sample: node_id={top_nodes[0]}, smiles='{top_nodes[1]}'")
        
        # Test neighbors endpoint (if we have neighbors)
        if node_count > 0:
            test_node_id = top_nodes[0]  # Use first top-level node
            response = client.get(f"/neighbors/{test_node_id}/0")
            
            if response.status_code == 200:
                data = response.json()
                neighbors = data["neighbors"]
                neighbor_count = data["neighbor_count"]
                
                # Check format
                assert len(neighbors) == neighbor_count * 2
                if neighbor_count > 0:
                    assert isinstance(neighbors[0], int)  # node_id
                    assert isinstance(neighbors[1], str)  # smiles
                    
                print(f"✓ Neighbors format correct: {neighbor_count} neighbors")
                if neighbor_count > 0:
                    print(f"  Sample: node_id={neighbors[0]}, smiles='{neighbors[1]}'")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_missing_database():
    """Test server behavior when database is not provided."""
    try:
        from rad.hnsw_server import HNSWServerApp
    except ImportError:
        print("❌ Cannot import HNSWServerApp - skipping test")
        return
    
    hnsw = create_test_hnsw(10)
    
    # Create server without database
    server_app = HNSWServerApp(
        hnsw=hnsw,
        database_path=None,
        debug=True
    )
    
    assert server_app.database_path is None
    print("✓ Server handles missing database gracefully")
    
    # Test SMILES lookup returns empty
    import asyncio
    
    async def test_no_smiles():
        smiles_map = await server_app._get_smiles_batch([0, 1, 2])
        assert smiles_map == {}
        return True
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(test_no_smiles())
    loop.close()
    
    print("✓ Missing database returns empty SMILES map")

def run_all_tests():
    """Run all tests."""
    print("🧪 Running SMILES Integration Tests")
    print("=" * 50)
    
    try:
        test_database_creation()
        test_hnsw_server_with_database()
        test_api_response_format()
        test_missing_database()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    run_all_tests()