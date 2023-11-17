import numpy as np
import pytest

from src.vectorizer import Vectorizer


@pytest.fixture
def vectorizer():
    return Vectorizer()


def test_vectorize_output_type(vectorizer):
    text = ["Hello world", "Testing sentence"]
    vectors = vectorizer.vectorize(text)
    assert all(isinstance(v, np.ndarray) for v in vectors)


def test_vectorize_output_size(vectorizer):
    text = ["Hello world", "Testing sentence"]
    vectors = vectorizer.vectorize(text)
    assert len(vectors) == len(text)


def test_vectorize_empty_input(vectorizer):
    text = []
    vectors = vectorizer.vectorize(text)
    assert vectors == []


def test_dimensionality_reduction_output_type(vectorizer):
    text = ["Hello world", "Testing sentence"]
    vectors = vectorizer.vectorize(text)
    reduced = vectorizer.dimensionality_reduction(vectors, 2)
    assert isinstance(reduced, np.ndarray)


def test_dimensionality_reduction_output_shape(vectorizer):
    text = ["Hello world", "Testing sentence"]
    vectors = vectorizer.vectorize(text)
    components = 2
    reduced = vectorizer.dimensionality_reduction(vectors, components)
    assert reduced.shape == (len(text), components)
