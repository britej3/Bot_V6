import pytest
from backend.src.services.feature_flag_service import FeatureFlagService

# This test will fail because the service doesn't exist yet
def test_feature_flag_service_loads_flags():
    service = FeatureFlagService()
    flag = service.get_flag("GRAPH_INTEGRATION_ENABLED")
    assert flag is not None
    assert flag.enabled is True
