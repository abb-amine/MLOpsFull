import numpy as np

from madewithml import analyze


class TestPlotTagDistribution:
    def test_returns_figure(self, sample_df):
        fig = analyze.plot_tag_distribution(sample_df)
        assert fig is not None

    def test_figure_type(self, sample_df):
        fig = analyze.plot_tag_distribution(sample_df)
        assert "Figure" in type(fig).__name__


class TestPlotWordCloud:
    def test_returns_figure(self, sample_df):
        fig = analyze.plot_wordcloud(sample_df, tag="computer-vision")
        assert fig is not None

    def test_figure_type(self, sample_df):
        fig = analyze.plot_wordcloud(sample_df, tag="computer-vision")
        assert "Figure" in type(fig).__name__


class TestFindLowConfidence:
    def test_no_low_confidence(self, class_to_index):
        y_true = np.array([0, 0, 0])
        y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.95, 0.05]])
        result = analyze.find_low_confidence(y_true, y_prob, class_to_index, "computer-vision", threshold=0.7)
        assert len(result) == 0

    def test_some_low_confidence(self, class_to_index):
        y_true = np.array([0, 0, 0])
        y_prob = np.array([[0.9, 0.1], [0.4, 0.6], [0.95, 0.05]])
        result = analyze.find_low_confidence(y_true, y_prob, class_to_index, "computer-vision", threshold=0.5)
        assert len(result) == 1
        assert result[0]["true"] == "computer-vision"


class TestFindLabelIssues:
    def test_returns_indices(self, sample_labels, sample_probabilities):
        result = analyze.find_label_issues(sample_labels, sample_probabilities)
        assert isinstance(result, np.ndarray)
        assert len(result) <= len(sample_labels)
