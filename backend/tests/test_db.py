from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.db.models import Base, Session as SessionModel
from app.db.database import get_db
import pytest


@pytest.fixture
def db_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    return engine


def test_session_model_create(db_engine):
    with Session(db_engine) as db:
        s = SessionModel(filename="test.csv", row_count=100,
                         profile_path="/tmp/p.json", data_path="/tmp/d.parquet")
        db.add(s)
        db.commit()
        db.refresh(s)
    assert s.id is not None
    assert s.filename == "test.csv"
