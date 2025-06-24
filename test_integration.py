#!/usr/bin/env python3
"""
Integration tests for the scalable RAD architecture.
Tests end-to-end functionality with real validation.
"""

import numpy as np
import time
from usearch.index import Index
from rad.traverser import RADTraverser

def create_test_data():
    """Create test data for integration testing"""
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
    
    scores = {key: np.random.uniform(0, 100) for key in keys}
    scoring_fn = lambda key: scores[key]
    
    return hnsw, keys, scoring_fn

def test_single_worker_traversal():
    """Test single worker traversal functionality"""
    print("Testing single worker traversal...")
    hnsw, keys, scoring_fn = create_test_data()
    traverser = RADTraverser(hnsw=hnsw, scoring_fn=scoring_fn)
    
    try:
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
        
        # Validate scoring consistency
        for key, score in results[:5]:
            expected_score = scoring_fn(key)
            assert abs(score - expected_score) < 1e-6, f"Score mismatch for key {key}"
        
        print("  ✓ Single worker test passed")
        
    finally:
        traverser.shutdown()

def test_multi_worker_traversal():
    """Test multi-worker traversal functionality"""
    print("\nTesting multi-worker traversal...")
    hnsw, keys, scoring_fn = create_test_data()
    traverser = RADTraverser(hnsw=hnsw, scoring_fn=scoring_fn)
    
    try:
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
        
        # Validate result consistency 
        print("  Validating scoring consistency...")
        for key, score in results[:5]:
            expected_score = scoring_fn(key)
            assert abs(score - expected_score) < 1e-6, f"Score mismatch for key {key}"
        
        print("  ✓ Multi-worker test passed")
        
    finally:
        traverser.shutdown()

def test_service_lifecycle():
    """Test service initialization and shutdown"""
    print("\nTesting service lifecycle management...")
    hnsw, keys, scoring_fn = create_test_data()
    traverser = RADTraverser(hnsw=hnsw, scoring_fn=scoring_fn)
    
    # Test service health
    print("  Checking service initialization and health...")
    assert traverser.is_initialized, "Services should be initialized"
    assert traverser.hnsw_service.is_healthy(), "HNSW service should be healthy"
    
    # Test graceful shutdown
    print("  Testing graceful shutdown...")
    traverser.shutdown()
    assert not traverser.is_running, "Traverser should not be running after shutdown"
    print("  ✓ Service lifecycle test passed")

def test_termination_conditions():
    """Test different termination conditions"""
    print("\nTesting termination conditions...")
    hnsw, keys, scoring_fn = create_test_data()
    
    # Test n_to_score termination
    print("  Testing n_to_score termination (target: 10)...")
    traverser1 = RADTraverser(hnsw=hnsw, scoring_fn=scoring_fn)
    try:
        traverser1.prime()
        traverser1.traverse(n_workers=1, n_to_score=10)
        results1 = list(traverser1.scored_set)
        assert len(results1) >= 10, "Should score at least target number"
        print(f"  ✓ Score-based termination: scored {len(results1)} molecules")
    finally:
        traverser1.shutdown()
    
    # Test timeout termination
    print("  Testing timeout termination (2 seconds)...")
    traverser2 = RADTraverser(hnsw=hnsw, scoring_fn=scoring_fn)
    try:
        traverser2.prime()
        start_time = time.time()
        traverser2.traverse(n_workers=1, timeout=2)
        end_time = time.time()
        assert end_time - start_time <= 3, "Should respect timeout"
        print(f"  ✓ Timeout termination: completed in {end_time - start_time:.2f}s")
    finally:
        traverser2.shutdown()
    print("  ✓ Termination conditions test passed")

def test_concurrent_safety():
    """Test that concurrent operations are safe"""
    print("\nTesting concurrent safety with 4 workers...")
    hnsw, keys, scoring_fn = create_test_data()
    traverser = RADTraverser(hnsw=hnsw, scoring_fn=scoring_fn)
    
    try:
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
        traverser.shutdown()

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
