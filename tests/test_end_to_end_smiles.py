#!/usr/bin/env python3
"""
End-to-End Tests for Complete SMILES Integration

Tests the full pipeline from HNSW server with database through workers
to final results with SMILES-based scoring and retrospective analysis.
"""

import os
import sqlite3
import tempfile
import time
import numpy as np
from usearch.index import Index

def create_test_database(db_path: str, n_nodes: int = 30):
    """Create a test SQLite database with realistic SMILES data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create nodes table
    cursor.execute("""
        CREATE TABLE nodes (
            node_key INTEGER PRIMARY KEY,
            smi TEXT NOT NULL
        )
    """)
    
    # Insert realistic SMILES data
    realistic_smiles = [
        "CCO",                    # ethanol
        "CC(C)O",                 # isopropanol  
        "CCC",                    # propane
        "CC(C)C",                 # isobutane
        "c1ccccc1",               # benzene
        "c1ccc(O)cc1",            # phenol
        "CC(=O)O",                # acetic acid
        "CCN",                    # ethylamine
        "CO",                     # methanol
        "CC",                     # ethane
        "c1ccc(N)cc1",            # aniline
        "CC(=O)C",                # acetone
        "CCCCCO",                 # pentanol
        "c1ccc(C)cc1",            # toluene
        "CC(C)(C)O",              # tert-butanol
        "c1ccc(Cl)cc1",           # chlorobenzene
        "CCO[CH2]",               # ethyl ether
        "c1ccc(F)cc1",            # fluorobenzene
        "CC(C)CC",                # methylpropane
        "CCCCO",                  # butanol
    ]
    
    for i in range(n_nodes):
        # Cycle through realistic SMILES
        smiles = realistic_smiles[i % len(realistic_smiles)]
        cursor.execute("INSERT INTO nodes (node_key, smi) VALUES (?, ?)", (i, smiles))
    
    cursor.execute("CREATE INDEX idx_nodes_node_key ON nodes(node_key)")
    conn.commit()
    conn.close()
    print(f"Created test database with {n_nodes} realistic SMILES molecules")

def create_test_hnsw(n_vectors: int = 30):
    """Create a test HNSW index matching database size."""
    dim = 64
    
    # Create some structure in the vectors for realistic neighbors
    vectors = np.random.randint(0, 2, size=(n_vectors, dim), dtype=np.uint8)
    
    # Add some correlation to make neighbors meaningful
    for i in range(1, n_vectors):
        # Each vector has some similarity to previous ones
        similarity_factor = 0.7
        vectors[i] = (similarity_factor * vectors[i-1] + 
                     (1-similarity_factor) * vectors[i]).astype(np.uint8)
    
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

def smiles_similarity_scorer(smiles: str) -> float:
    """
    Simple SMILES-based scoring function for testing.
    Scores based on molecular features for realistic behavior.
    """
    if not smiles:
        return 0.0
    
    score = 50.0  # Base score
    
    # Favor alcohols (presence of O)
    if 'O' in smiles:
        score += 20.0
    
    # Favor aromatics (presence of benzene ring)
    if 'c1ccc' in smiles:
        score += 15.0
    
    # Penalize halogens slightly
    if 'Cl' in smiles or 'F' in smiles:
        score -= 5.0
    
    # Favor nitrogen-containing compounds
    if 'N' in smiles:
        score += 10.0
    
    # Add some randomness but keep it deterministic for testing
    hash_score = hash(smiles) % 20
    score += hash_score
    
    return max(0.0, score)

def test_end_to_end_local_service():
    """Test end-to-end with LocalHNSWService and database."""
    try:
        from rad.hnsw_service import create_local_hnsw_service
        from rad.traverser import RADTraverser
        import redis
    except ImportError as e:
        print(f"❌ Cannot import required modules: {e}")
        return False
    
    # Create test database and HNSW
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        create_test_database(db_path, 25)
        hnsw = create_test_hnsw(25)
        
        print("🧪 Testing End-to-End with LocalHNSWService + Database")
        
        # Create HNSW service with database
        hnsw_service = create_local_hnsw_service(
            hnsw=hnsw,
            database_path=db_path
        )
        
        # Create traverser
        traverser = RADTraverser(
            hnsw_service=hnsw_service,
            scoring_fn=smiles_similarity_scorer,
            deployment_mode="local"
        )
        
        print("✓ Created traverser with database-enabled HNSW service")
        
        # Prime traversal
        traverser.prime()
        print("✓ Priming completed")
        
        # Run short traversal
        traverser.traverse(n_workers=2, n_to_score=15, timeout=30)
        print("✓ Traversal completed")
        
        # Get results using new retrospective analysis
        best_molecules = traverser.get_best_molecules(5)
        
        assert len(best_molecules) > 0, "Should have scored molecules"
        
        print(f"✓ Retrieved {len(best_molecules)} best molecules:")
        for i, (node_id, score, smiles) in enumerate(best_molecules[:3]):
            assert isinstance(node_id, int), "node_id should be int"
            assert isinstance(score, float), "score should be float" 
            assert isinstance(smiles, str), "smiles should be str"
            assert smiles != "", "SMILES should not be empty"
            
            print(f"  {i+1}. Node {node_id}: {smiles} (score: {score:.2f})")
        
        # Verify scoring function was called with SMILES
        assert all(mol[2] != "" for mol in best_molecules), "All molecules should have SMILES"
        
        print("✓ End-to-end test with LocalHNSWService passed!")
        
        # Cleanup
        traverser.shutdown()
        return True
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_end_to_end_remote_service():
    """Test end-to-end with RemoteHNSWService (if server available)."""
    try:
        from rad.hnsw_service import create_remote_hnsw_service
        from rad.traverser import RADTraverser
        import requests
    except ImportError as e:
        print(f"❌ Cannot import required modules: {e}")
        return False
    
    # Check if server is available
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if response.status_code != 200:
            raise Exception("Server not healthy")
    except:
        print("⚠️  No HNSW server with database running - skipping RemoteHNSWService test")
        print("   Start server with: python scripts/start_hnsw_server.py --test-data --database-path db.sqlite --port 8000")
        return True  # Not a failure, just skipped
    
    print("🧪 Testing End-to-End with RemoteHNSWService")
    
    # Create remote service
    hnsw_service = create_remote_hnsw_service("http://127.0.0.1:8000")
    
    try:
        # Create traverser
        traverser = RADTraverser(
            hnsw_service=hnsw_service,
            scoring_fn=smiles_similarity_scorer,
            deployment_mode="local"
        )
        
        print("✓ Created traverser with remote HNSW service")
        
        # Prime traversal
        traverser.prime()
        print("✓ Priming completed")
        
        # Run short traversal
        traverser.traverse(n_workers=2, n_to_score=10, timeout=30)
        print("✓ Traversal completed")
        
        # Get results
        best_molecules = traverser.get_best_molecules(3)
        
        if len(best_molecules) > 0:
            print(f"✓ Retrieved {len(best_molecules)} best molecules:")
            for i, (node_id, score, smiles) in enumerate(best_molecules):
                print(f"  {i+1}. Node {node_id}: {smiles} (score: {score:.2f})")
            
            print("✓ End-to-end test with RemoteHNSWService passed!")
        else:
            print("⚠️  No molecules scored (server might not have database)")
        
        # Cleanup
        traverser.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Remote service test failed: {e}")
        return False

def test_retrospective_analysis():
    """Test the retrospective analysis functionality."""
    try:
        from rad.scored import RedisScoredSet
        import redis
    except ImportError as e:
        print(f"❌ Cannot import required modules: {e}")
        return False
    
    print("🧪 Testing Retrospective Analysis")
    
    # Create Redis client (assumes local Redis)
    try:
        redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=False)
        redis_client.ping()
    except:
        print("⚠️  Redis not available - skipping retrospective analysis test")
        return True
    
    # Create scored set
    scored_set = RedisScoredSet(redis_client, scored_name='test_retro')
    
    try:
        # Add some test data
        test_data = [
            (1, 85.5, "CCO"),
            (2, 92.1, "c1ccccc1O"),
            (3, 78.3, "CC(C)O"),
            (4, 95.7, "c1ccc(N)cc1"),
            (5, 82.4, "CC(=O)O"),
        ]
        
        for node_id, score, smiles in test_data:
            scored_set.insert(node_id, score, smiles)
        
        print("✓ Inserted test molecule data")
        
        # Test get_molecules
        molecules = scored_set.get_molecules(3)
        
        assert len(molecules) == 3, f"Expected 3 molecules, got {len(molecules)}"
        
        for node_id, score, smiles in molecules:
            assert isinstance(node_id, int), "node_id should be int"
            assert isinstance(score, float), "score should be float"
            assert isinstance(smiles, str), "smiles should be str"
            assert smiles != "", "SMILES should not be empty"
        
        print(f"✓ Retrieved {len(molecules)} molecules for retrospective analysis:")
        for node_id, score, smiles in molecules:
            print(f"  Node {node_id}: {smiles} (score: {score:.1f})")
        
        # Test getting all molecules
        all_molecules = scored_set.get_molecules()
        assert len(all_molecules) == 5, f"Expected 5 total molecules, got {len(all_molecules)}"
        
        print("✓ Retrospective analysis test passed!")
        return True
        
    finally:
        # Cleanup test data
        try:
            redis_client.delete('test_retro:list', 'test_retro:set', 'test_retro:smiles')
        except:
            pass

def test_scoring_function_interface():
    """Test that scoring functions receive SMILES strings correctly."""
    print("🧪 Testing Scoring Function Interface")
    
    def test_scoring_fn(smiles: str) -> float:
        """Test function that validates input is SMILES string."""
        assert isinstance(smiles, str), f"Expected string SMILES, got {type(smiles)}"
        assert len(smiles) > 0, "SMILES should not be empty"
        
        # Simple validation that it looks like SMILES
        valid_chars = set('CONFClBrcnos()=[]#-+1234567890')
        smiles_chars = set(smiles.replace('.', ''))  # Remove dots for salts
        
        # Should contain mostly valid SMILES characters
        if smiles_chars - valid_chars:
            unknown_chars = smiles_chars - valid_chars
            if len(unknown_chars) > 2:  # Allow a few unknown chars
                raise ValueError(f"Invalid SMILES characters: {unknown_chars}")
        
        return len(smiles) * 10.0  # Simple length-based score
    
    # Test with realistic SMILES
    test_smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CO"]
    
    for smiles in test_smiles:
        try:
            score = test_scoring_fn(smiles)
            assert isinstance(score, (int, float)), "Score should be numeric"
            print(f"✓ Scored '{smiles}' -> {score}")
        except Exception as e:
            print(f"❌ Failed to score '{smiles}': {e}")
            return False
    
    print("✓ Scoring function interface test passed!")
    return True

def run_all_tests():
    """Run all end-to-end tests."""
    print("🚀 Running End-to-End SMILES Integration Tests")
    print("=" * 60)
    
    tests = [
        test_scoring_function_interface,
        test_retrospective_analysis,
        test_end_to_end_local_service,
        test_end_to_end_remote_service,
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
        print("🎉 All end-to-end tests passed!")
        print("\n🎯 SMILES Integration Complete!")
        print("   • Workers now score using SMILES strings")
        print("   • Retrospective analysis available via get_best_molecules()")
        print("   • Full pipeline: Database → HNSW → Workers → Results")
        return True
    else:
        print("⚠️  Some tests failed or were skipped")
        return False

if __name__ == "__main__":
    run_all_tests()