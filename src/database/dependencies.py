"""
Database dependencies for FastAPI dependency injection.
"""
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from .manager import DatabaseManager
except ImportError:
    from database.manager import DatabaseManager

# Global database manager instance - will be set from main.py
db_manager: DatabaseManager = None

def set_database_manager(manager: DatabaseManager):
    """Set the global database manager instance."""
    global db_manager
    db_manager = manager

# Dependency to get a database session
async def get_db():
    if db_manager is None:
        raise RuntimeError("Database manager not initialized. Call set_database_manager() first.")

    async with db_manager.get_session() as session:
        yield session