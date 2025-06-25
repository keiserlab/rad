#!/usr/bin/env python3
"""
Simple example showing how to use RADTraverser with remote HNSW service.

This example demonstrates the clean, service-oriented approach where you can
do complete RAD traversal using only a remote HNSW endpoint.
"""

import numpy as np
from rad.hnsw_service import create_remote_hnsw_service
from rad.traverser import RADTraverser, create_remote_traverser
import logging

logging.basicConfig(level=logging.ERROR, format='%(name)s - %(levelname)s - %(message)s')

def main():
    """Example of using remote HNSW service with RADTraverser."""
    
    print("🚀 Remote HNSW Service Example")
    print("=" * 40)
    
    # Step 1: Connect to remote HNSW service
    print("1. Connecting to remote HNSW service...")
    try:
        remote_service = create_remote_hnsw_service(
            "http://127.0.0.1:8000",  # Assumes HNSW server is running
            is_default=True
        )
        print("   ✓ Connected successfully!")
        
        # Get HNSW info from remote service
        hnsw_info = remote_service.get_hnsw_info()
        print(f"   📊 Remote HNSW: {hnsw_info['size']} vectors, max_level={hnsw_info['max_level']}")
        
    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        print("\n💡 To run this example:")
        print("   1. Start HNSW server: python scripts/start_hnsw_server.py --test-data --port 8000")
        print("   2. Run this example: python example_remote_hnsw.py")
        return
    
    # Step 2: Create scoring function
    print("\n2. Setting up scoring function...")
    # Simple scoring function - in practice this would score molecules
    scores = {key: np.random.uniform(0, 100) for key in range(100)}
    scoring_fn = lambda key: scores.get(key, 0)
    print("   ✓ Scoring function ready")
    
    # Step 3: Create traverser with remote HNSW service
    print("\n3. Creating RAD traverser with remote HNSW...")
    
    # Method 1: Direct service specification (clean approach)
    traverser = RADTraverser(
        hnsw_service=remote_service,
        scoring_fn=scoring_fn
    )
    print("   ✓ Traverser created with remote HNSW service")
    
    # Alternative Method 2: Convenience function
    # traverser = create_remote_traverser(
    #     "http://127.0.0.1:8000",
    #     scoring_fn=scoring_fn
    # )
    
    try:
        # Step 4: Prime traversal 
        print("\n4. Priming traversal...")
        traverser.prime()
        print("   ✓ Traversal primed with top-level nodes")
        
        # Step 5: Run distributed traversal
        print("\n5. Running distributed traversal...")
        print("   🔄 Scoring molecules with remote HNSW...")
        
        traverser.traverse(n_workers=2, n_to_score=20)
        
        # Step 6: Get results
        results = list(traverser.scored_set)
        print(f"   ✓ Scored {len(results)} molecules using remote HNSW!")
        
        # Show sample results
        print(f"\n📈 Sample results:")
        for i, (key, score) in enumerate(results[:5]):
            print(f"   {i+1}. Molecule {key}: score = {score:.2f}")
        
        print(f"\n🎉 Success! Remote HNSW traversal completed:")
        print(f"   • Remote HNSW service handled all neighbor queries")
        print(f"   • No local HNSW index required")
        print(f"   • Scored {len(results)} molecules total")
        print(f"   • Used {2} distributed workers")
        
    except Exception as e:
        print(f"   ❌ Traversal failed: {e}")
        
    finally:
        # Step 7: Cleanup
        print("\n6. Cleaning up...")
        traverser.shutdown()
        remote_service.shutdown()
        print("   ✓ All services shut down cleanly")

if __name__ == "__main__":
    main()