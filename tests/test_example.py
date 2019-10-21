import pytest

from multiview.example.example import example_function


def test_example_function():
    """
    Test that example function returns correct value.
    """
    assert example_function() == "param"
    assert example_function("hello") == "hello"
