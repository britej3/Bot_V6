import pytest
import json
from pathlib import Path
from backend.src.services.feature_flag_service import FeatureFlagService

def test_feature_flag_service_handles_missing_file(tmp_path: Path):
    config_file = tmp_path / "non_existent_file.json"

    # Monkeypatch the CONFIG_PATH in the service
    import backend.src.services.feature_flag_service as ff_service
    ff_service.CONFIG_PATH = config_file

    service = FeatureFlagService()
    assert service.get_all_flags() == []

def test_feature_flag_service_handles_empty_file(tmp_path: Path):
    config_file = tmp_path / "empty.json"
    config_file.write_text("{}")

    # Monkeypatch the CONFIG_PATH in the service
    import backend.src.services.feature_flag_service as ff_service
    ff_service.CONFIG_PATH = config_file

    service = FeatureFlagService()
    assert service.get_all_flags() == []
