"""Small database initialization / check helpers.

Schema itself is owned by Alembic migrations; these helpers only:

- verify the database is reachable (``check_connection``),
- ensure the ``vector`` extension exists (``ensure_vector_extension``),
- verify the ``assets`` table exists, i.e. migrations were applied
  (``init_db``).

Run ``python -m app.database.init_db`` from ``backend/`` before starting
the backend to fail fast with a helpful message instead of a cryptic
connection error later.
"""

from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from app.database.session import SessionLocal, engine


def check_connection() -> bool:
    """Return True if the database accepts a connection."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


def ensure_vector_extension() -> None:
    """Create the pgvector extension if it is missing."""
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def vector_available() -> bool:
    """Return True if the pgvector extension is installed."""
    with SessionLocal() as db:
        row = db.execute(
            text(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            )
        ).first()
        return row is not None


def init_db() -> None:
    """Prepare the database: extension + migrated-table check."""
    if not check_connection():
        raise RuntimeError(
            "Cannot connect to PostgreSQL. Is the database running? "
            "Start it with `docker compose up -d` from the project root, "
            "then run `alembic upgrade head` from backend/."
        )
    ensure_vector_extension()
    tables = inspect(engine).get_table_names()
    if "assets" not in tables:
        raise RuntimeError(
            "Connected, but the `assets` table is missing. "
            "Run `alembic upgrade head` from backend/ to apply migrations."
        )


if __name__ == "__main__":
    init_db()
    print(f"Database OK: vector extension={vector_available()}, tables include assets.")
