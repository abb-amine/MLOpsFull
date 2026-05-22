import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def sample_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "created_on": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "title": [
                "Comparison between YOLO and RCNN",
                "Transformer for NLP tasks",
                "Graph Neural Networks overview",
            ],
            "description": [
                "Bringing theory to experiment is cool.",
                "Using pretrained models for text classification.",
                "A collection of important graph embedding papers.",
            ],
            "tag": ["computer-vision", "natural-language-processing", "graph-learning"],
        }
    )


@pytest.fixture(scope="session")
def class_to_index():
    return {
        "computer-vision": 0,
        "graph-learning": 1,
        "natural-language-processing": 2,
        "other": 3,
        "reinforcement-learning": 4,
    }


@pytest.fixture(scope="session")
def sample_labels():
    rng = np.random.RandomState(42)
    return rng.randint(0, 5, size=100)


@pytest.fixture(scope="session")
def sample_predictions():
    rng = np.random.RandomState(42)
    return rng.randint(0, 5, size=100)


@pytest.fixture(scope="session")
def sample_probabilities():
    rng = np.random.RandomState(42)
    probs = rng.dirichlet(np.ones(5), size=100)
    return probs.astype(np.float64)


@pytest.fixture(scope="session")
def sample_numpy_arrays():
    return np.array([[1, 2, 3], [4, 5], [6]], dtype=object)
