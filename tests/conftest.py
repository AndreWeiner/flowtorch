"""Pytest configuration for the flowTorch test suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests requiring external data, processes, or devices",
    )
