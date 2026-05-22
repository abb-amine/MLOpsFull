import datetime
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

import pandas as pd


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and record.extra_data:
            log_data["extra"] = record.extra_data
        if record.exc_info and record.exc_info[0]:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def log_timing(logger: Optional[logging.Logger] = None):
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            extra = {"duration_seconds": round(elapsed, 4)}
            logger.info(f"{func.__name__} completed", extra={"extra_data": extra})
            return result

        return wrapper

    return decorator


def log_data_shape(
    data: Any,
    name: str = "",
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = logging.getLogger("madewithml.log_utils")
    if isinstance(data, pd.DataFrame):
        extra = {"rows": len(data), "columns": list(data.columns), "shape": list(data.shape)}
    elif isinstance(data, (list, tuple)):
        extra = {"length": len(data)}
    elif hasattr(data, "count"):
        extra = {"count": data.count()}
    else:
        extra = {"length": len(data)} if hasattr(data, "__len__") else {}
    label = f" {name}" if name else ""
    logger.info(f"Data{label}: {json.dumps(extra)}", extra={"extra_data": extra})


def log_config(config: Dict, name: str = "", logger: Optional[logging.Logger] = None) -> None:
    if logger is None:
        logger = logging.getLogger("madewithml.log_utils")
    label = f" {name}" if name else ""
    logger.info(f"Config{label}: {json.dumps(config, default=str)}", extra={"extra_data": config})


def log_class_distribution(series: pd.Series, name: str = "", logger: Optional[logging.Logger] = None) -> None:
    if logger is None:
        logger = logging.getLogger("madewithml.log_utils")
    counts = series.value_counts().to_dict()
    extra = {"distribution": counts, "total": int(series.sum()) if series.dtype.kind in "iuf" else len(series)}
    label = f" {name}" if name else ""
    logger.info(f"Distribution{label}: {json.dumps(extra)}", extra={"extra_data": extra})
