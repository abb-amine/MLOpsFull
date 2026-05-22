import numpy as np

from madewithml import data


class TestCleanText:
    def test_lowercase(self):
        result = data.clean_text("Hello World")
        assert result == "hello world"

    def test_remove_stopwords(self):
        result = data.clean_text("this is a test with the word and")
        assert "this" not in result
        assert "test" in result

    def test_remove_punctuation(self):
        result = data.clean_text("hello, world! (test)")
        assert "," not in result
        assert "!" not in result
        assert "(" not in result

    def test_remove_multiple_spaces(self):
        result = data.clean_text("hello    world")
        assert result == "hello world"

    def test_strip_whitespace(self):
        result = data.clean_text("  hello world  ")
        assert result == "hello world"

    def test_remove_links(self):
        result = data.clean_text("check http://example.com out")
        assert "http" not in result

    def test_remove_non_alphanumeric(self):
        result = data.clean_text("hello#$%world")
        assert result == "hello world"

    def test_empty_string(self):
        result = data.clean_text("")
        assert result == ""

    def test_only_stopwords(self):
        result = data.clean_text("the and or but")
        assert result == ""


class TestPreprocess:
    def test_preprocess_output_keys(self, sample_df, class_to_index):
        result = data.preprocess(sample_df, class_to_index)
        assert "ids" in result
        assert "masks" in result
        assert "targets" in result

    def test_preprocess_removes_columns(self, sample_df, class_to_index):
        result = data.preprocess(sample_df, class_to_index)
        assert len(result["targets"]) == len(sample_df)

    def test_preprocess_label_encoding(self, sample_df, class_to_index):
        result = data.preprocess(sample_df, class_to_index)
        expected = np.array(
            [
                class_to_index["computer-vision"],
                class_to_index["natural-language-processing"],
                class_to_index["graph-learning"],
            ]
        )
        np.testing.assert_array_equal(result["targets"], expected)


class TestCustomPreprocessor:
    def test_fit_creates_mappings(self):
        import ray

        ds = ray.data.from_items(
            [
                {"tag": "a"},
                {"tag": "b"},
                {"tag": "a"},
                {"tag": "c"},
            ]
        )
        preprocessor = data.CustomPreprocessor()
        preprocessor.fit(ds)
        assert "a" in preprocessor.class_to_index
        assert "b" in preprocessor.class_to_index
        assert "c" in preprocessor.class_to_index
        assert preprocessor.index_to_class[preprocessor.class_to_index["a"]] == "a"
