#!/usr/bin/env python3
"""
Integration tests for the scalable RAD architecture.
Tests end-to-end functionality with real validation.
"""

import numpy as np
import time
import tempfile
import sqlite3
import os
from usearch.index import Index
from rad.traverser import RADTraverser
from rad.hnsw_service import create_local_hnsw_service

def create_test_database(db_path: str, n_nodes: int = 100):
    """Create a test SQLite database with node_key -> SMILES mapping."""
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

def create_test_data():
    """Create test data for integration testing with SMILES support"""
    n_vectors = 100
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
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    create_test_database(db_path, n_vectors)
    
    # Create SMILES-based scoring function
    smiles_scores = {}
    for i in range(n_vectors):
        test_smiles = ["CCO", "CCC", "CC(C)C", "c1ccccc1", "CC(=O)O", "CCN", "CO", "CC"]
        base_smiles = test_smiles[i % len(test_smiles)]
        smiles = f"{base_smiles}.{i}" if i > 0 else base_smiles
        smiles_scores[smiles] = np.random.uniform(0, 100)
    
    def scoring_fn(smiles: str) -> float:
        return smiles_scores.get(smiles, 50.0)  # Default score if SMILES not found
    
    return hnsw, db_path, scoring_fn

def test_single_worker_traversal():
    """Test single worker traversal functionality"""
    print("Testing single worker traversal...")
    hnsw, db_path, scoring_fn = create_test_data()
    
    try:
        hnsw_service = create_local_hnsw_service(hnsw, database_path=db_path)
        traverser = RADTraverser(hnsw_service=hnsw_service, scoring_fn=scoring_fn)
        
        print("  Priming traversal with top-level nodes...")
        traverser.prime()
        
        print("  Starting single worker traversal (target: 30 molecules)...")
        start_time = time.time()
        traverser.traverse(n_workers=1, n_to_score=30)
        end_time = time.time()
        
        results = list(traverser.scored_set)
        print(f"  Completed in {end_time - start_time:.2f}s, scored {len(results)} molecules")
        
        # Validate basic functionality
        assert len(results) >= 30, f"Expected at least 30 results, got {len(results)}"
        assert end_time - start_time < 10, "Traversal took too long"
        
        # Validate result format
        print("  Validating result format and consistency...")
        for key, score in results[:5]:
            assert isinstance(key, int), f"Key should be int, got {type(key)}"
            assert isinstance(score, (int, float)), f"Score should be numeric, got {type(score)}"
            assert 0 <= score <= 100, f"Score should be 0-100, got {score}"
        
        print("  ✓ Single worker test passed")
        
    finally:
        try:
            traverser.shutdown()
        except:
            pass
        # Clean up database file
        try:
            os.unlink(db_path)
        except:
            pass

def test_multi_worker_traversal():
    """Test multi-worker traversal functionality"""
    print("\nTesting multi-worker traversal...")
    hnsw, db_path, scoring_fn = create_test_data()
    
    try:
        hnsw_service = create_local_hnsw_service(hnsw, database_path=db_path)
        traverser = RADTraverser(hnsw_service=hnsw_service, scoring_fn=scoring_fn)
        
        print("  Priming traversal with top-level nodes...")
        traverser.prime()
        
        print("  Starting 3-worker traversal (target: 30 molecules)...")
        start_time = time.time()
        traverser.traverse(n_workers=3, n_to_score=30)
        end_time = time.time()
        
        results = list(traverser.scored_set)
        print(f"  Completed in {end_time - start_time:.2f}s, scored {len(results)} molecules")
        
        # Validate multi-worker results
        assert len(results) >= 30, f"Expected at least 30 results, got {len(results)}"
        assert end_time - start_time < 15, "Multi-worker traversal took too long"
        
        # Validate no duplicate scoring (this was the main bug we fixed)
        print("  Checking for duplicate scoring (the critical race condition test)...")
        scored_keys = set(key for key, score in results)
        assert len(scored_keys) == len(results), "Duplicate keys found in results"
        print(f"  ✓ No duplicates found: {len(results)} results, {len(scored_keys)} unique keys")
        
        print("  ✓ Multi-worker test passed")
        
    finally:
        try:
            traverser.shutdown()
        except:
            pass
        # Clean up database file
        try:
            os.unlink(db_path)
        except:
            pass

def test_service_lifecycle():
    """Test service initialization and shutdown"""
    print("\nTesting service lifecycle management...")
    hnsw, db_path, scoring_fn = create_test_data()
    
    try:
        hnsw_service = create_local_hnsw_service(hnsw, database_path=db_path)
        traverser = RADTraverser(hnsw_service=hnsw_service, scoring_fn=scoring_fn)
        
        # Test service health
        print("  Checking service initialization and health...")
        assert traverser.is_initialized, "Services should be initialized"
        assert traverser.hnsw_service.is_healthy(), "HNSW service should be healthy"
        
        # Test graceful shutdown
        print("  Testing graceful shutdown...")
        traverser.shutdown()
        assert not traverser.is_running, "Traverser should not be running after shutdown"
        print("  ✓ Service lifecycle test passed")
    finally:
        # Clean up database file
        try:
            os.unlink(db_path)
        except:
            pass

def test_termination_conditions():
    """Test different termination conditions"""
    print("\nTesting termination conditions...")
    hnsw, db_path, scoring_fn = create_test_data()
    db_path2 = None
    
    try:
        # Test n_to_score termination
        print("  Testing n_to_score termination (target: 10)...")
        hnsw_service1 = create_local_hnsw_service(hnsw, database_path=db_path)
        traverser1 = RADTraverser(hnsw_service=hnsw_service1, scoring_fn=scoring_fn)
        
        traverser1.prime()
        traverser1.traverse(n_workers=1, n_to_score=10)
        results1 = list(traverser1.scored_set)
        assert len(results1) >= 10, "Should score at least target number"
        print(f"  ✓ Score-based termination: scored {len(results1)} molecules")
        traverser1.shutdown()
        
        # Test timeout termination - need a separate database
        print("  Testing timeout termination (2 seconds)...")
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db2:
            db_path2 = tmp_db2.name
        create_test_database(db_path2, 100)
        
        hnsw_service2 = create_local_hnsw_service(hnsw, database_path=db_path2)
        traverser2 = RADTraverser(hnsw_service=hnsw_service2, scoring_fn=scoring_fn)
        
        traverser2.prime()
        start_time = time.time()
        traverser2.traverse(n_workers=1, timeout=2)
        end_time = time.time()
        assert end_time - start_time <= 3, "Should respect timeout"
        print(f"  ✓ Timeout termination: completed in {end_time - start_time:.2f}s")
        traverser2.shutdown()
        
        print("  ✓ Termination conditions test passed")
        
    finally:
        # Clean up database files
        for path in [db_path, db_path2]:
            if path:
                try:
                    os.unlink(path)
                except:
                    pass

def test_concurrent_safety():
    """Test that concurrent operations are safe"""
    print("\nTesting concurrent safety with 4 workers...")
    hnsw, db_path, scoring_fn = create_test_data()
    
    try:
        hnsw_service = create_local_hnsw_service(hnsw, database_path=db_path)
        traverser = RADTraverser(hnsw_service=hnsw_service, scoring_fn=scoring_fn)
        
        print("  Priming traversal...")
        traverser.prime()
        
        # Multiple workers should not interfere with each other
        print("  Starting high-concurrency test (4 workers, target: 50 molecules)...")
        start_time = time.time()
        traverser.traverse(n_workers=4, n_to_score=50)
        end_time = time.time()
        results = list(traverser.scored_set)
        print(f"  Completed in {end_time - start_time:.2f}s, scored {len(results)} molecules")
        
        # Validate no corruption
        assert len(results) >= 50, "Should complete successfully with multiple workers"
        
        # Check for data consistency (most important test for race conditions)
        print("  Validating data consistency under high concurrency...")
        scored_keys = [key for key, score in results]
        assert len(set(scored_keys)) == len(scored_keys), "No duplicate keys should exist"
        print(f"  ✓ No race conditions: {len(results)} results, all unique")
        print("  ✓ Concurrent safety test passed")
        
    finally:
        try:
            traverser.shutdown()
        except:
            pass
        # Clean up database file
        try:
            os.unlink(db_path)
        except:
            pass

if __name__ == "__main__":
    print("Running RAD integration tests...")
    print("=" * 50)
    
    # Run atomic tests
    test_single_worker_traversal()
    test_multi_worker_traversal() 
    test_service_lifecycle()
    test_termination_conditions()
    test_concurrent_safety()
    
    print("\n" + "=" * 50)
    print("🎉 All integration tests passed!")
