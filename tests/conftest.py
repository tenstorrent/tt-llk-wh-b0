# conftest.py
def pytest_addoption(parser):
    parser.addoption(
        "--length", action="store", default=3, type=int, help="Length of the math kernel operations"
    )
