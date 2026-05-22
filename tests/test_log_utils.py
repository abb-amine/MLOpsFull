import json
import logging
import time

import pandas as pd

from madewithml.log_utils import JSONFormatter, log_class_distribution, log_config, log_data_shape, log_timing


class TestJSONFormatter:
    def setup_method(self):
        self.formatter = JSONFormatter()
        self.logger = logging.getLogger("test_json")
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def test_format_basic(self):
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            "test_module",
            42,
            "hello world",
            (),
            None,
        )
        output = self.formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["message"] == "hello world"
        assert data["module"] == "test_module"
        assert data["line"] == 42
        assert "timestamp" in data

    def test_format_with_extra(self):
        record = self.logger.makeRecord(
            self.logger.name,
            logging.WARNING,
            "mod",
            10,
            "with extra",
            (),
            None,
        )
        record.extra_data = {"key": "value"}
        output = self.formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "WARNING"
        assert data["extra"]["key"] == "value"

    def test_format_no_extra(self):
        record = self.logger.makeRecord(
            self.logger.name,
            logging.ERROR,
            "mod",
            1,
            "no extra",
            (),
            None,
        )
        output = self.formatter.format(record)
        data = json.loads(output)
        assert "extra" not in data


class TestLogTiming:
    def test_decorator_logs_duration(self, caplog):
        caplog.set_level(logging.INFO)
        test_logger = logging.getLogger("test_timing")
        test_logger.handlers.clear()
        test_logger.addHandler(logging.StreamHandler())
        test_logger.setLevel(logging.INFO)

        @log_timing(test_logger)
        def dummy():
            time.sleep(0.01)
            return 42

        result = dummy()
        assert result == 42


class TestLogDataShape:
    def test_dataframe(self, caplog):
        caplog.set_level(logging.INFO)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        log_data_shape(df, name="test")
        assert any("test" in record.message for record in caplog.records)
        assert any("rows" in record.message for record in caplog.records)

    def test_list(self, caplog):
        caplog.set_level(logging.INFO)
        log_data_shape([1, 2, 3], name="list_data")
        assert any("length" in record.message for record in caplog.records)


class TestLogConfig:
    def test_log_config_dict(self, caplog):
        caplog.set_level(logging.INFO)
        log_config({"lr": 0.001, "epochs": 10}, name="train")
        assert any("Config train" in record.message for record in caplog.records)


class TestLogClassDistribution:
    def test_distribution(self, caplog):
        caplog.set_level(logging.INFO)
        series = pd.Series(["a", "b", "a", "c", "a", "b"])
        log_class_distribution(series, name="tags")
        assert any("Distribution tags" in record.message for record in caplog.records)
