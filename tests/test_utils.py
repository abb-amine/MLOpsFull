import os
import tempfile

import numpy as np

from madewithml import utils


class TestSetSeeds:
    def test_set_seeds(self):
        utils.set_seeds(seed=42)
        a = np.random.randn(10)
        utils.set_seeds(seed=42)
        b = np.random.randn(10)
        np.testing.assert_array_almost_equal(a, b)


class TestPadArray:
    def test_pad_uneven_rows(self, sample_numpy_arrays):
        result = utils.pad_array(sample_numpy_arrays)
        assert result.shape == (3, 3)
        assert result[0].tolist() == [1, 2, 3]
        assert result[1].tolist() == [4, 5, 0]
        assert result[2].tolist() == [6, 0, 0]

    def test_pad_empty(self):
        arr = np.array([[]], dtype=object)
        result = utils.pad_array(arr)
        assert result.shape == (1, 0)

    def test_pad_single_element(self):
        arr = np.array([[42]], dtype=object)
        result = utils.pad_array(arr)
        assert result.shape == (1, 1)
        assert result[0][0] == 42


class TestDictToList:
    def test_dict_to_list_basic(self):
        data = {
            "epoch": [0, 1, 2],
            "train_loss": [0.5, 0.3, 0.1],
            "val_loss": [0.6, 0.4, 0.2],
        }
        result = utils.dict_to_list(data, keys=["epoch", "train_loss", "val_loss"])
        assert len(result) == 3
        assert result[0] == {"epoch": 0, "train_loss": 0.5, "val_loss": 0.6}
        assert result[1] == {"epoch": 1, "train_loss": 0.3, "val_loss": 0.4}

    def test_dict_to_list_subset_keys(self):
        data = {"epoch": [0, 1], "train_loss": [0.5, 0.3], "val_loss": [0.6, 0.4]}
        result = utils.dict_to_list(data, keys=["epoch"])
        assert result == [{"epoch": 0}, {"epoch": 1}]


class TestSaveLoadDict:
    def test_save_and_load_dict(self):
        d = {"name": "test", "value": 42, "nested": {"a": 1}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            utils.save_dict(d, path)
            loaded = utils.load_dict(path)
            assert loaded == d
        finally:
            os.remove(path)

    def test_save_dict_creates_parent_dirs(self):
        d = {"key": "value"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "subdir", "output.json")
            utils.save_dict(d, path)
            assert os.path.exists(path)
            loaded = utils.load_dict(path)
            assert loaded == d
