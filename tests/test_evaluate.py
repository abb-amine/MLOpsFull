import numpy as np

from madewithml import evaluate


class TestGetOverallMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        metrics = evaluate.get_overall_metrics(y_true, y_pred)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["num_samples"] == 4

    def test_worst_prediction(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([3, 2, 1, 0])
        metrics = evaluate.get_overall_metrics(y_true, y_pred)
        assert metrics["precision"] < 0.5
        assert metrics["recall"] < 0.5
        assert metrics["f1"] < 0.5
        assert metrics["num_samples"] == 4

    def test_single_class(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        metrics = evaluate.get_overall_metrics(y_true, y_pred)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_mixed(self, sample_labels, sample_predictions):
        metrics = evaluate.get_overall_metrics(sample_labels, sample_predictions)
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert metrics["num_samples"] == len(sample_labels)


class TestGetPerClassMetrics:
    def test_per_class_structure(self, sample_labels, sample_predictions, class_to_index):
        metrics = evaluate.get_per_class_metrics(sample_labels, sample_predictions, class_to_index)
        for tag in class_to_index:
            assert tag in metrics
            assert "precision" in metrics[tag]
            assert "recall" in metrics[tag]
            assert "f1" in metrics[tag]
            assert "num_samples" in metrics[tag]

    def test_perfect_per_class(self, class_to_index):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        metrics = evaluate.get_per_class_metrics(y_true, y_pred, class_to_index)
        for tag in class_to_index:
            assert metrics[tag]["f1"] == 1.0

    def test_sorted_by_f1(self, sample_labels, sample_predictions, class_to_index):
        metrics = evaluate.get_per_class_metrics(sample_labels, sample_predictions, class_to_index)
        f1_scores = [metrics[tag]["f1"] for tag in metrics]
        assert all(f1_scores[i] >= f1_scores[i + 1] for i in range(len(f1_scores) - 1))
