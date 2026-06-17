import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.main import app
from app.db.database import get_db
from app.db.models import Base


@pytest.fixture
def sample_csv_bytes() -> bytes:
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "Department": ["HR", "HR", "Engineering", "engineering", "Sales"],
        "Satisfaction": [4, 2, 5, 3, 4],
        "Comments": ["Great place", "Too many meetings", "", "Good team", "Loves it"],
        "Salary": [50000, 60000, 80000, 75000, 55000],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def sample_excel_bytes() -> bytes:
    df = pd.DataFrame({
        "Department": ["HR", "Engineering", "Sales"],
        "Headcount": [10, 25, 15],
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="Q1")
    return buf.getvalue()


@pytest.fixture
def tmp_db(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import sessionmaker
    TestSession = sessionmaker(bind=engine)
    db = TestSession()
    yield db
    db.close()


@pytest.fixture
def client(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")

    def override_get_db():
        yield tmp_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
