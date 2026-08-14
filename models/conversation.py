import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine

load_dotenv()

class Base(DeclarativeBase):
    pass

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

database_url = os.getenv("DATABASE_URL", "sqlite:///conversations.db")
# If using Postgres, convert postgresql:// to postgresql+psycopg:// to use psycopg v3
# instead of the default psycopg2 (which isn't installed)
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
engine = create_engine(database_url)

try:
    Base.metadata.create_all(engine)
except Exception:
    # Allow app to start even if database is unreachable (e.g., network issues during dev,
    # or migrations haven't run yet in production). The error will surface when a route
    # tries to access the database.
    pass
