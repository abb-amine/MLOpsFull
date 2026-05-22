def test_core_imports():
    from madewithml import config, data, evaluate, log_utils, models, predict, train, tune, utils

    assert config.ROOT_DIR.exists()
    assert hasattr(data, "load_data")
    assert hasattr(data, "clean_text")
    assert hasattr(models, "FinetunedLLM")
    assert hasattr(train, "train_model")
    assert hasattr(tune, "tune_models")
    assert hasattr(evaluate, "evaluate")
    assert hasattr(predict, "TorchPredictor")
    assert hasattr(utils, "set_seeds")
    assert hasattr(log_utils, "JSONFormatter")
    assert hasattr(log_utils, "log_timing")
