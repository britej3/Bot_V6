import asyncio
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import asynccontextmanager

from src.database.models import Base # Import Base from your models file

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database Manager with SQLAlchemy for async operations, connection pooling, and transaction management.
    """
    def __init__(self, db_url: str = "sqlite+aiosqlite:///./test.db"):
        self.db_url = db_url
        self.engine: Optional[AsyncEngine] = None
        self.async_session_factory: Optional[sessionmaker] = None
        logger.info(f"DatabaseManager initialized with URL: {self.db_url}")

    async def connect(self):
        """Establishes connection to the database and creates tables if they don't exist."""
        if self.engine is None:
            # For SQLite, don't use pooling parameters
            if "sqlite" in self.db_url:
                self.engine = create_async_engine(self.db_url, echo=False)
            else:
                self.engine = create_async_engine(self.db_url, echo=False, pool_size=10, max_overflow=20)
            self.async_session_factory = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database connected and tables ensured (SQLAlchemy).")

    async def disconnect(self):
        """Closes the database connection."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.async_session_factory = None
            logger.info("Database disconnected (SQLAlchemy).")

    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Provides an async session for database operations."""
        if self.async_session_factory is None:
            raise RuntimeError("DatabaseManager not connected. Call connect() first.")

        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Session rollback due to error: {e}")
            raise
        finally:
            await session.close()

    async def fetch_one(self, query: Any, params: Optional[Dict] = None) -> Optional[Dict]:
        """Fetches a single row using SQLAlchemy Core or ORM query."""
        async with self.get_session() as session:
            result = await session.execute(query, params)
            row = result.first()
            return row._asdict() if row else None

    async def fetch_all(self, query: Any, params: Optional[Dict] = None) -> List[Dict]:
        """Fetches all rows using SQLAlchemy Core or ORM query."""
        async with self.get_session() as session:
            result = await session.execute(query, params)
            return [row._asdict() for row in result.all()]

    async def execute(self, query: Any, params: Optional[Dict] = None) -> Any:
        """Executes a DML statement using SQLAlchemy Core or ORM query."""
        async with self.get_session() as session:
            result = await session.execute(query, params)
            return result.rowcount # Or other relevant result like lastrowid
