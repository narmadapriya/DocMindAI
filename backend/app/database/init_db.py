from sqlalchemy import text

from app.database.connection import engine


def check_database():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        print("PostgreSQL Connected Successfully")