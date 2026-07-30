"""Pytest configuration for the flowTorch test suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests that require external datasets or other large fixtures",
    )
