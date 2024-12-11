import pytest
import os
import glob

def run_all_tests(test_files):
    result = pytest.main(test_files)

if __name__ == '__main__':
    test_files = glob.glob('test_*.py')
    os.system("tt-smi -r 0")
    
    exit_code = run_all_tests(test_files)

    if exit_code == pytest.ExitCode.OK:
        print("All tests passed!")
    elif exit_code == pytest.ExitCode.TESTS_FAILED:
        print("Some tests failed.")
    else:
        print("An unexpected error occurred.")
