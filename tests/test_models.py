import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from madewithml.models import FinetunedLLM


@pytest.fixture(scope="module")
def model():
    from transformers import BertModel

    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = FinetunedLLM(llm=llm, dropout_p=0.5, embedding_dim=llm.config.hidden_size, num_classes=5)
    model.eval()
    return model


@pytest.mark.requires_scibert
class TestFinetunedLLM:
    def test_model_initialization(self, model):
        assert model.dropout_p == 0.5
        assert model.embedding_dim == 768
        assert model.num_classes == 5

    def test_forward_pass(self, model):
        batch = {
            "ids": torch.randint(0, 1000, (2, 50)),
            "masks": torch.ones(2, 50, dtype=torch.int32),
        }
        output = model(batch)
        assert output.shape == (2, 5)

    def test_predict(self, model):
        batch = {
            "ids": torch.randint(0, 1000, (2, 50)),
            "masks": torch.ones(2, 50, dtype=torch.int32),
        }
        predictions = model.predict(batch)
        assert predictions.shape == (2,)
        assert all(0 <= p < 5 for p in predictions)

    def test_predict_proba(self, model):
        batch = {
            "ids": torch.randint(0, 1000, (2, 50)),
            "masks": torch.ones(2, 50, dtype=torch.int32),
        }
        probs = model.predict_proba(batch)
        assert probs.shape == (2, 5)
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(2))

    def test_save_and_load(self, model):
        with tempfile.TemporaryDirectory() as tmpdir:
            dp = Path(tmpdir)
            model.save(dp)
            assert (dp / "args.json").exists()
            assert (dp / "model.pt").exists()
            loaded = FinetunedLLM.load(dp / "args.json", dp / "model.pt")
            assert loaded.dropout_p == model.dropout_p
            assert loaded.embedding_dim == model.embedding_dim
            assert loaded.num_classes == model.num_classes
