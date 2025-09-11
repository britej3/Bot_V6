"""
Minimal test to verify database functionality without full app import
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends
from database.dependencies import get_db, set_database_manager
from database.manager import DatabaseManager


def test_database_connection():
    """Test database connection works"""
    async def setup_and_test():
        # Initialize database manager
        db_manager = DatabaseManager(db_url="sqlite+aiosqlite:///./test.db")
        await db_manager.connect()
        set_database_manager(db_manager)

        # Create a minimal FastAPI app for testing
        app = FastAPI()

        @app.get("/test-db")
        async def test_db_endpoint(db=Depends(get_db)):
            return {"status": "Database connected"}

        # Test with client
        with TestClient(app) as client:
            response = client.get("/test-db")
            assert response.status_code == 200
            assert response.json() == {"status": "Database connected"}

    # Run the async test
    asyncio.run(setup_and_test())


if __name__ == "__main__":
    test_database_connection()
    print("Database test passed!")