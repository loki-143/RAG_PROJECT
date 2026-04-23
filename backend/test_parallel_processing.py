"""
Test script to verify parallel processing implementation.

This script tests:
1. Sequential vs parallel chunking produces identical results
2. Parallel processing is faster than sequential
3. Worker count configuration works correctly
4. Error handling in parallel mode
"""

import time
import logging
from indexer import RepositoryIndexer
from storage import ChunkStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_parallel_vs_sequential():
    """Test that parallel and sequential produce identical results."""
    logger.info("=" * 60)
    logger.info("TEST 1: Parallel vs Sequential - Result Consistency")
    logger.info("=" * 60)
    
    # Use a small test repo (this repo itself)
    test_repo = "https://github.com/psf/requests"  # Well-known small repo
    
    indexer = RepositoryIndexer()
    
    # Test sequential
    logger.info("\n--- Sequential Processing ---")
    start = time.time()
    chunks_seq, meta_seq = indexer.index_repository(
        test_repo,
        force_reindex=True,
        use_parallel=False
    )
    time_seq = time.time() - start
    logger.info(f"Sequential: {len(chunks_seq)} chunks in {time_seq:.2f}s")
    
    # Test parallel
    logger.info("\n--- Parallel Processing ---")
    start = time.time()
    chunks_par, meta_par = indexer.index_repository(
        test_repo,
        force_reindex=True,
        use_parallel=True
    )
    time_par = time.time() - start
    logger.info(f"Parallel: {len(chunks_par)} chunks in {time_par:.2f}s")
    
    # Compare results
    logger.info("\n--- Results Comparison ---")
    logger.info(f"Chunk count match: {len(chunks_seq) == len(chunks_par)}")
    logger.info(f"Sequential time: {time_seq:.2f}s")
    logger.info(f"Parallel time: {time_par:.2f}s")
    
    if time_par < time_seq:
        speedup = time_seq / time_par
        logger.info(f"✓ Speedup: {speedup:.2f}x faster")
    else:
        logger.info(f"⚠ Parallel was slower (repo may be too small)")
    
    # Verify chunk IDs match (order may differ)
    seq_ids = {c.chunk_id for c in chunks_seq}
    par_ids = {c.chunk_id for c in chunks_par}
    
    if seq_ids == par_ids:
        logger.info("✓ Chunk IDs match perfectly")
    else:
        missing_in_par = seq_ids - par_ids
        missing_in_seq = par_ids - seq_ids
        logger.error(f"✗ Chunk ID mismatch!")
        logger.error(f"  Missing in parallel: {len(missing_in_par)}")
        logger.error(f"  Missing in sequential: {len(missing_in_seq)}")
    
    # Cleanup
    chunk_store = ChunkStore()
    chunk_store.delete_index(test_repo)
    
    return len(chunks_seq) == len(chunks_par) and seq_ids == par_ids


def test_worker_configuration():
    """Test that worker count configuration works."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Worker Configuration")
    logger.info("=" * 60)
    
    # Test with different worker counts
    for workers in [1, 2, 4]:
        logger.info(f"\n--- Testing with {workers} workers ---")
        indexer = RepositoryIndexer(max_workers=workers)
        logger.info(f"Configured workers: {indexer.max_workers}")
        assert indexer.max_workers == workers, f"Worker count mismatch: {indexer.max_workers} != {workers}"
    
    # Test auto-detection
    logger.info(f"\n--- Testing auto-detection (None) ---")
    indexer = RepositoryIndexer(max_workers=None)
    logger.info(f"Auto-detected workers: {indexer.max_workers}")
    assert indexer.max_workers >= 1, "Auto-detection failed"
    
    logger.info("✓ Worker configuration test passed")
    return True


def test_error_handling():
    """Test error handling in parallel mode."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Error Handling")
    logger.info("=" * 60)
    
    indexer = RepositoryIndexer()
    
    # Test with invalid repo (should handle gracefully)
    try:
        logger.info("\n--- Testing with invalid repo URL ---")
        chunks, meta = indexer.index_repository(
            "https://github.com/nonexistent/repo12345",
            use_parallel=True
        )
        logger.error("✗ Should have raised an error")
        return False
    except Exception as e:
        logger.info(f"✓ Correctly raised error: {type(e).__name__}")
        return True


def run_all_tests():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("PARALLEL PROCESSING TEST SUITE")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Consistency
    try:
        results['consistency'] = test_parallel_vs_sequential()
    except Exception as e:
        logger.error(f"Test 1 failed with error: {e}")
        results['consistency'] = False
    
    # Test 2: Configuration
    try:
        results['configuration'] = test_worker_configuration()
    except Exception as e:
        logger.error(f"Test 2 failed with error: {e}")
        results['configuration'] = False
    
    # Test 3: Error handling
    try:
        results['error_handling'] = test_error_handling()
    except Exception as e:
        logger.error(f"Test 3 failed with error: {e}")
        results['error_handling'] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
    else:
        logger.info("✗ SOME TESTS FAILED")
    logger.info("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
