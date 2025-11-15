import os
from datetime import datetime

import config


def test_project_directories_exist():
    """Importing config should eagerly create the key project folders."""
    for path in (config.DATA_DIR, config.MODEL_DIR, config.OUTPUT_DIR, config.LOG_DIR):
        assert os.path.isdir(path), f"Expected directory to exist: {path}"


def test_data_config_date_window():
    """The configured start/end dates should cover the expected 3-year window."""
    start = datetime.strptime(config.DATA_CONFIG["start_date"], "%Y-%m-%d").date()
    end = datetime.strptime(config.DATA_CONFIG["end_date"], "%Y-%m-%d").date()

    assert (end - start).days == 1095
    assert end <= datetime.utcnow().date()
