#!/usr/bin/env python3
"""
Clean, atomic tests for the HNSW Service abstraction layer.
"""

import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from usearch.index import Index
from rad.hnsw_service import LocalHNSWService, ServiceRegistry, create_local_hnsw_service, service_registry

def create_test_hnsw(n_vectors=100, dim=64):
    """Create a simple HNSW for testing"""
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

def test_local_hnsw_service_basic():
    """Test basic LocalHNSWService functionality"""
    hnsw = create_test_hnsw()
    service = LocalHNSWService(hnsw)
    
    try:
        # Test service info
        info = service.get_service_info()
        assert info['service_type'] == 'LocalHNSWService'
        assert info['status'] == 'running'
        
        # Test health check
        assert service.is_healthy() is True
        
        # Test get_neighbors
        neighbors = service.get_neighbors(0, 0)
        assert neighbors is not None
        assert len(neighbors) > 0
        
        # Test get_top_level_nodes
        top_nodes = service.get_top_level_nodes()
        assert top_nodes is not None
        assert len(top_nodes) > 0
        
    finally:
        service.shutdown()

def test_concurrent_requests():
    """Test concurrent requests to LocalHNSWService"""
    hnsw = create_test_hnsw()
    service = LocalHNSWService(hnsw)
    
    def worker_requests(worker_id, num_requests):
        """Worker function that makes multiple requests"""
        results = []
        for i in range(num_requests):
            try:
                neighbors = service.get_neighbors(worker_id * 10 + i, 0)
                results.append({
                    'worker_id': worker_id,
                    'request_id': i,
                    'neighbors_count': len(neighbors),
                    'success': True
                })
            except Exception as e:
                results.append({
                    'worker_id': worker_id,
                    'request_id': i,
                    'error': str(e),
                    'success': False
                })
        return results
    
    try:
        # Test with multiple threads making concurrent requests
        num_workers = 5
        requests_per_worker = 10
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker_requests, worker_id, requests_per_worker)
                for worker_id in range(num_workers)
            ]
            
            all_results = []
            for future in futures:
                worker_results = future.result()
                all_results.extend(worker_results)
        
        # Validate results
        successful_requests = [r for r in all_results if r['success']]
        failed_requests = [r for r in all_results if not r['success']]
        
        expected_total = num_workers * requests_per_worker
        
        assert len(successful_requests) == expected_total, f"Expected {expected_total} successful requests, got {len(successful_requests)}"
        assert len(failed_requests) == 0, f"Got {len(failed_requests)} failed requests"
        
        # Validate service state
        info = service.get_service_info()
        assert info['error_count'] == 0, f"Service reported {info['error_count']} errors"
        
    finally:
        service.shutdown()

def test_service_registry():
    """Test the ServiceRegistry for service discovery"""
    registry = ServiceRegistry()
    hnsw = create_test_hnsw()
    
    service1 = LocalHNSWService(hnsw)
    service2 = LocalHNSWService(hnsw)
    
    try:
        # Register services
        registry.register_service("primary", service1, is_default=True)
        registry.register_service("secondary", service2)
        
        # Test service retrieval
        primary = registry.get_service("primary")
        secondary = registry.get_service("secondary")
        default = registry.get_service()  # Should get default (primary)
        
        assert primary is service1
        assert secondary is service2
        assert default is service1
        
        # Test service listing
        services_info = registry.list_services()
        assert "primary" in services_info
        assert "secondary" in services_info
        
        # Test operations on retrieved services
        neighbors1 = primary.get_neighbors(0, 0)
        neighbors2 = secondary.get_neighbors(0, 0)
        
        assert neighbors1 is not None
        assert neighbors2 is not None
        
    finally:
        registry.shutdown_all()

def test_convenience_function():
    """Test the convenience function for creating services"""
    hnsw = create_test_hnsw()
    
    # Clear any existing services
    service_registry.shutdown_all()
    
    # Create service using convenience function
    service = create_local_hnsw_service(hnsw)
    
    try:
        # Should be registered as default
        default_service = service_registry.get_service()
        assert default_service is service
        
        # Test that it works
        neighbors = default_service.get_neighbors(0, 0)
        assert neighbors is not None
        
        info = default_service.get_service_info()
        assert info['service_type'] == 'LocalHNSWService'
        
    finally:
        service_registry.shutdown_all()

def test_error_handling():
    """Test error handling and edge cases"""
    hnsw = create_test_hnsw()
    service = LocalHNSWService(hnsw, response_timeout=2.0)
    
    try:
        # Test normal operation first
        neighbors = service.get_neighbors(0, 0)
        assert neighbors is not None
        
        # Test shutdown and subsequent requests
        service.shutdown()
        
        # Service should report as not healthy
        assert service.is_healthy() is False
        
        # Requests should fail after shutdown
        try:
            service.get_neighbors(0, 0)
            assert False, "Request should have failed after shutdown"
        except RuntimeError:
            pass  # Expected error
        
    except Exception as e:
        # Make sure we clean up even if test fails
        try:
            service.shutdown()
        except:
            pass
        raise

def test_performance_metrics():
    """Test performance monitoring and metrics"""
    hnsw = create_test_hnsw()
    service = LocalHNSWService(hnsw)
    
    try:
        # Make several requests
        num_requests = 20
        
        for i in range(num_requests):
            neighbors = service.get_neighbors(i % 10, 0)
            assert neighbors is not None
        
        # Check metrics
        info = service.get_service_info()
        
        assert info['request_count'] >= num_requests
        assert info['error_count'] == 0
        assert info['error_rate'] == 0.0
        assert info['uptime_seconds'] > 0
        
    finally:
        service.shutdown()

if __name__ == "__main__":
    test_local_hnsw_service_basic()
    test_concurrent_requests()
    test_service_registry()
    test_convenience_function()
    test_error_handling()
    test_performance_metrics()
    print("All HNSW service tests passed!")