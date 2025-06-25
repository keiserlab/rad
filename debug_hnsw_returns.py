#!/usr/bin/env python3
"""
Debug script to see what usearch HNSW methods return
"""

import numpy as np
import sys
import os
sys.path.append('.')

# Import usearch (adjust path as needed)
try:
    import usearch.index as usearch
except ImportError:
    print("Could not import usearch")
    sys.exit(1)

def create_test_hnsw():
    """Create a small test HNSW index"""
    print("Creating test HNSW index...")
    
    # Create random vectors
    n_vectors = 100
    dim = 128
    vectors = np.random.random((n_vectors, dim)).astype(np.float32)
    
    # Create HNSW index
    hnsw = usearch.Index(ndim=dim, metric='cos')
    
    # Add vectors
    for i, vector in enumerate(vectors):
        hnsw.add(i, vector)
    
    print(f"Created HNSW with {len(hnsw)} vectors")
    return hnsw

def debug_methods(hnsw):
    """Debug the HNSW methods to see what they return"""
    
    print("\n" + "="*50)
    print("DEBUGGING HNSW METHODS")
    print("="*50)
    
    # Test get_neighbors
    print("\n1. Testing get_neighbors(1, 0):")
    try:
        neighbors = hnsw.get_neighbors(1, 0)
        print(f"  Raw return: {neighbors}")
        print(f"  Type: {type(neighbors)}")
        print(f"  Length: {len(neighbors) if hasattr(neighbors, '__len__') else 'No __len__'}")
        
        if hasattr(neighbors, '__iter__'):
            print("  Converting to list...")
            neighbors_list = list(neighbors)
            print(f"  List: {neighbors_list}")
            print(f"  List types: {[type(x) for x in neighbors_list[:5]]}")  # First 5 types
        
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test get_top_level_nodes
    print("\n2. Testing get_top_level_nodes():")
    try:
        top_nodes = hnsw.get_top_level_nodes()
        print(f"  Raw return: {top_nodes}")
        print(f"  Type: {type(top_nodes)}")
        print(f"  Length: {len(top_nodes) if hasattr(top_nodes, '__len__') else 'No __len__'}")
        
        if hasattr(top_nodes, '__iter__'):
            print("  Converting to list...")
            top_nodes_list = list(top_nodes)
            print(f"  List: {top_nodes_list}")
            print(f"  List types: {[type(x) for x in top_nodes_list[:5]]}")  # First 5 types
        
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test HNSW properties
    print("\n3. Testing HNSW properties:")
    try:
        print(f"  max_level: {hnsw.max_level} (type: {type(hnsw.max_level)})")
        print(f"  size: {len(hnsw)} (type: {type(len(hnsw))})")
        print(f"  connectivity: {hnsw.connectivity} (type: {type(hnsw.connectivity)})")
        print(f"  dtype: {hnsw.dtype} (type: {type(hnsw.dtype)})")
        print(f"  ndim: {hnsw.ndim} (type: {type(hnsw.ndim)})")
        print(f"  capacity: {hnsw.capacity} (type: {type(hnsw.capacity)})")
        print(f"  memory_usage: {hnsw.memory_usage} (type: {type(hnsw.memory_usage)})")
        print(f"  multi: {hnsw.multi} (type: {type(hnsw.multi)})")
    except Exception as e:
        print(f"  ERROR: {e}")

def main():
    print("HNSW Debug Script")
    print("="*50)
    
    # Create test HNSW
    hnsw = create_test_hnsw()
    
    # Debug the methods
    debug_methods(hnsw)
    
    print("\n" + "="*50)
    print("DEBUG COMPLETE")

if __name__ == "__main__":
    main()