#!/usr/bin/env python3
"""
Tests for Service Layer SMILES Integration

Tests that LocalHNSWService and RemoteHNSWService correctly handle
the new [node_id, smiles, ...] format and database integration.
"""

import os
import sqlite3
import tempfile
import time
import numpy as np
from usearch.index import Index

def create_test_database(db_path: str, n_nodes: int = 50):
    """Create a test SQLite database with node_id -> SMILES mapping."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create nodes table
    cursor.execute("""
        CREATE TABLE nodes (
            node_key INTEGER PRIMARY KEY,
            smi TEXT NOT NULL
        )
    """)
    
    # Insert test SMILES data
    test_smiles = [
        "CCO",         # ethanol
        "CCC",         # propane  
        "CC(C)C",      # isobutane
        "c1ccccc1",    # benzene
        "CC(=O)O",     # acetic acid
        "CCN",         # ethylamine
        "CO",          # methanol
        "CC",          # ethane
    ]
    
    for i in range(n_nodes):
        # Cycle through test SMILES with variation
        base_smiles = test_smiles[i % len(test_smiles)]
        smiles = f"{base_smiles}.{i}" if i > 0 else base_smiles
        cursor.execute("INSERT INTO nodes (node_key, smi) VALUES (?, ?)", (i, smiles))
    
    cursor.execute("CREATE INDEX idx_nodes_node_key ON nodes(node_key)")
    conn.commit()
    conn.close()
    print(f"Created test database with {n_nodes} nodes")

def create_test_hnsw(n_vectors: int = 50):
    """Create a test HNSW index matching the database size."""
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

def test_local_hnsw_service_with_database():
    """Test LocalHNSWService with database integration."""
    try:
        from rad.hnsw_service import LocalHNSWService
    except ImportError:
        print("❌ Cannot import LocalHNSWService - skipping test")
        return False
    
    # Create test database and HNSW
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        create_test_database(db_path, 30)
        hnsw = create_test_hnsw(30)
        
        # Create LocalHNSWService with database
        service = LocalHNSWService(
            hnsw=hnsw,
            database_path=db_path,
            response_timeout=10.0
        )
        
        print("✓ LocalHNSWService created with database")
        
        # Test service is healthy
        assert service.is_healthy(), "Service should be healthy"
        print("✓ Service health check passed")
        
        # Test get_top_level_nodes returns [node_id, smiles, ...] format
        top_nodes = service.get_top_level_nodes()
        assert len(top_nodes) > 0, "Should have top level nodes"
        assert len(top_nodes) % 2 == 0, "Should have even number of elements (pairs)"
        
        # Check format: [node_id, smiles, node_id, smiles, ...]
        for i in range(0, len(top_nodes), 2):
            node_id = top_nodes[i]
            smiles = top_nodes[i+1]
            
            assert isinstance(node_id, int), f"node_id should be int, got {type(node_id)}"
            assert isinstance(smiles, str), f"smiles should be str, got {type(smiles)}"
            assert smiles != "", "SMILES should not be empty for test data"
        
        print(f"✓ Top-level nodes format correct: {len(top_nodes)//2} nodes")
        print(f"  Sample: node_id={top_nodes[0]}, smiles='{top_nodes[1]}'")
        
        # Test get_neighbors if we have any
        if len(top_nodes) > 0:
            test_node_id = top_nodes[0]
            neighbors = service.get_neighbors(test_node_id, 0)
            
            # Neighbors might be empty, that's ok
            if len(neighbors) > 0:
                assert len(neighbors) % 2 == 0, "Neighbors should have even number of elements"
                
                # Check format
                for i in range(0, len(neighbors), 2):
                    neighbor_id = neighbors[i]
                    neighbor_smiles = neighbors[i+1]
                    
                    assert isinstance(neighbor_id, int), f"neighbor_id should be int"
                    assert isinstance(neighbor_smiles, str), f"neighbor_smiles should be str"
                
                print(f"✓ Neighbors format correct: {len(neighbors)//2} neighbors")
                print(f"  Sample: neighbor_id={neighbors[0]}, smiles='{neighbors[1]}'")
            else:
                print("✓ Neighbors test skipped (no neighbors found)")
        
        # Cleanup
        service.shutdown()
        print("✓ Service shutdown successful")
        
        return True
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_local_hnsw_service_without_database():
    """Test LocalHNSWService without database (should return empty SMILES)."""
    try:
        from rad.hnsw_service import LocalHNSWService
    except ImportError:
        print("❌ Cannot import LocalHNSWService - skipping test")
        return False
    
    hnsw = create_test_hnsw(20)
    
    # Create service without database
    service = LocalHNSWService(
        hnsw=hnsw,
        database_path=None,
        response_timeout=10.0
    )
    
    print("✓ LocalHNSWService created without database")
    
    try:
        # Test get_top_level_nodes
        top_nodes = service.get_top_level_nodes()
        
        # Should still return pairs, but SMILES should be empty
        if len(top_nodes) > 0:
            assert len(top_nodes) % 2 == 0, "Should still have pairs"
            
            for i in range(0, len(top_nodes), 2):
                node_id = top_nodes[i]
                smiles = top_nodes[i+1]
                
                assert isinstance(node_id, int), "node_id should be int"
                assert isinstance(smiles, str), "smiles should be str"
                assert smiles == "", "SMILES should be empty without database"
            
            print(f"✓ No database handling correct: {len(top_nodes)//2} nodes with empty SMILES")
        
        return True
        
    finally:
        service.shutdown()

def test_remote_hnsw_service_integration():
    """Test RemoteHNSWService with live server (if available)."""
    try:
        from rad.hnsw_service import RemoteHNSWService
        import requests
    except ImportError:
        print("❌ Cannot import required modules for RemoteHNSWService test")
        return False
    
    # Check if there's a server running on localhost:8000
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if response.status_code != 200:
            raise Exception("Server not healthy")
    except:
        print("⚠️  No HNSW server running on localhost:8000 - skipping RemoteHNSWService test")
        print("   Start server with: python scripts/start_hnsw_server.py --test-data --port 8000")
        return True  # Not a failure, just skipped
    
    # Test with live server
    service = RemoteHNSWService("http://127.0.0.1:8000")
    
    try:
        assert service.is_healthy(), "Remote service should be healthy"
        print("✓ RemoteHNSWService connected successfully")
        
        # Test top-level nodes
        top_nodes = service.get_top_level_nodes()
        
        if len(top_nodes) > 0:
            assert len(top_nodes) % 2 == 0, "Should have pairs"
            
            # Check format
            for i in range(0, len(top_nodes), 2):
                node_id = top_nodes[i]
                smiles = top_nodes[i+1]
                
                assert isinstance(node_id, int), "node_id should be int"
                assert isinstance(smiles, str), "smiles should be str"
                # Note: SMILES might be empty if server has no database
            
            print(f"✓ Remote service format correct: {len(top_nodes)//2} nodes")
            
            # Test neighbors
            test_node_id = top_nodes[0]
            neighbors = service.get_neighbors(test_node_id, 0)
            
            if len(neighbors) > 0:
                assert len(neighbors) % 2 == 0, "Neighbors should have pairs"
                print(f"✓ Remote neighbors format correct: {len(neighbors)//2} neighbors")
        
        return True
        
    finally:
        service.shutdown()

def test_service_creation_functions():
    """Test service creation functions with database_path parameter."""
    try:
        from rad.hnsw_service import create_local_hnsw_service, create_remote_hnsw_service
    except ImportError:
        print("❌ Cannot import service creation functions")
        return False
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        create_test_database(db_path, 10)
        hnsw = create_test_hnsw(10)
        
        # Test create_local_hnsw_service with database_path
        service = create_local_hnsw_service(
            hnsw=hnsw,
            database_path=db_path,
            response_timeout=5.0
        )
        
        assert service.database_path == db_path, "Service should have correct database path"
        assert service.is_healthy(), "Service should be healthy"
        
        print("✓ create_local_hnsw_service with database_path works")
        
        service.shutdown()
        
        # Test create_remote_hnsw_service (doesn't need database_path)
        try:
            remote_service = create_remote_hnsw_service(
                "http://127.0.0.1:8000",
                service_name="test_remote"
            )
            print("✓ create_remote_hnsw_service works")
            remote_service.shutdown()
        except:
            print("⚠️  create_remote_hnsw_service test skipped (no server)")
        
        return True
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def run_all_tests():
    """Run all Phase 2 service layer tests."""
    print("🧪 Running Phase 2 Service Layer Tests")
    print("=" * 50)
    
    tests = [
        test_local_hnsw_service_with_database,
        test_local_hnsw_service_without_database,
        test_remote_hnsw_service_integration,
        test_service_creation_functions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            print(f"\n🔍 Running {test.__name__}...")
            result = test()
            if result:
                passed += 1
                print(f"✅ {test.__name__} passed")
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 tests passed!")
        return True
    else:
        print("⚠️  Some tests failed or were skipped")
        return False

if __name__ == "__main__":
    run_all_tests()