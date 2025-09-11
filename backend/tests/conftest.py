import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from backend.src.main import app
import json
from backend.src.services.feature_flag_service import CONFIG_PATH

@pytest.fixture(scope="function", autouse=True)
def reset_feature_flags():
    with open(CONFIG_PATH, "w") as f:
        json.dump(
            {"GRAPH_INTEGRATION_ENABLED": {"enabled": True, "description": "Enable/disable graph database integration."}},
            f,
            indent=2
        )
    yield
    with open(CONFIG_PATH, "w") as f:
        json.dump(
            {"GRAPH_INTEGRATION_ENABLED": {"enabled": True, "description": "Enable/disable graph database integration."}},
            f,
            indent=2
        )


@pytest_asyncio.fixture(scope="function")
async def client() -> AsyncClient:
    # The service is a singleton, we need to reload it to pick up the changes from the reset_feature_flags fixture
    from backend.src.services.feature_flag_service import feature_flag_service
    feature_flag_service.load_flags()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client