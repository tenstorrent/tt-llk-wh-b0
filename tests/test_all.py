import pytest
import os

def run_test_file(file_path):
    result = pytest.main([file_path, '--tb=short'])
    return result

if __name__ == '__main__':
    results = {}
    test_files = ['test_eltwise.py', 'test_multiple_kernels.py']  # Add your test files

    for test_file in test_files:
        os.system("/home/software/syseng/wh/tt-smi -wr 0")
        results[test_file] = run_test_file(test_file)

    total_passed = sum(1 for result in results.values() if result == 0)
    total_failed = len(test_files) - total_passed

    print(f"Total tests passed: {total_passed}")
    print(f"Total tests failed: {total_failed}")
