import pytest
from httpx import AsyncClient

# This test will fail because the endpoint doesn't exist yet
@pytest.mark.asyncio
async def test_get_feature_flags(client: AsyncClient):
    response = await client.get("/feature-flags")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
