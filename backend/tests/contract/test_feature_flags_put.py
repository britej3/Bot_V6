import pytest
from httpx import AsyncClient

# This test will fail because the endpoint doesn't exist yet
@pytest.mark.asyncio
async def test_update_feature_flag(client: AsyncClient):
    response = await client.put(
        "/feature-flags/GRAPH_INTEGRATION_ENABLED",
        json={"enabled": False},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "GRAPH_INTEGRATION_ENABLED"
    assert response.json()["enabled"] is False
