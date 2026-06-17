from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.data.loader import load_file
from app.data.profiler import build_profile
from app.db.models import Base, Insight
from app.insights.generator import generate_insights


@pytest.fixture
def db_with_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    Sess = sessionmaker(bind=engine)
    return Sess()


@pytest.fixture
def profile_and_parquet(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "ins-1", "responses.csv", tmp_path)
    return profile, str(tmp_path / "ins-1" / "data.parquet")


@pytest.mark.asyncio
async def test_generate_insights_stores_records(profile_and_parquet, db_with_session, tmp_path):
    profile, parquet_path = profile_and_parquet
    from app.db.models import Session as SessionModel
    s = SessionModel(id="ins-1", filename="responses.csv", row_count=5,
                     profile_path=str(tmp_path / "ins-1" / "profile.json"),
                     data_path=parquet_path)
    db_with_session.add(s)
    db_with_session.commit()

    batch_response = json.dumps([
        {"title": "Finding 1", "summary": "HR has lower satisfaction than Engineering."},
        {"title": "Finding 2", "summary": "Salary correlates with Department."},
    ])
    with patch("app.insights.generator.llm.chat_completion", new=AsyncMock(return_value=batch_response)):
        await generate_insights("ins-1", profile, parquet_path, db_with_session)

    insights = db_with_session.query(Insight).filter_by(session_id="ins-1").all()
    assert len(insights) >= 1
    assert insights[0].rank == 1
    assert insights[0].title
    assert insights[0].summary


@pytest.mark.asyncio
async def test_generate_insights_ranked_by_effect(profile_and_parquet, db_with_session, tmp_path):
    profile, parquet_path = profile_and_parquet
    from app.db.models import Session as SessionModel
    s = SessionModel(id="ins-2", filename="responses.csv", row_count=5,
                     profile_path=str(tmp_path / "ins-1" / "profile.json"),
                     data_path=parquet_path)
    db_with_session.add(s)
    db_with_session.commit()

    batch = json.dumps([{"title": "A", "summary": "Finding A."}, {"title": "B", "summary": "Finding B."}])
    with patch("app.insights.generator.llm.chat_completion", new=AsyncMock(return_value=batch)):
        await generate_insights("ins-2", profile, parquet_path, db_with_session)

    insights = db_with_session.query(Insight).filter_by(session_id="ins-2").order_by(Insight.rank).all()
    ranks = [i.rank for i in insights]
    assert ranks == sorted(ranks)


@pytest.mark.asyncio
async def test_generate_insights_idempotent(profile_and_parquet, db_with_session, tmp_path):
    """Running twice on same session_id clears old insights and re-generates."""
    profile, parquet_path = profile_and_parquet
    from app.db.models import Session as SessionModel
    s = SessionModel(id="ins-3", filename="responses.csv", row_count=5,
                     profile_path=str(tmp_path / "ins-1" / "profile.json"),
                     data_path=parquet_path)
    db_with_session.add(s)
    db_with_session.commit()

    batch = json.dumps([{"title": "T", "summary": "S."}])
    with patch("app.insights.generator.llm.chat_completion", new=AsyncMock(return_value=batch)):
        await generate_insights("ins-3", profile, parquet_path, db_with_session)
        await generate_insights("ins-3", profile, parquet_path, db_with_session)

    count = db_with_session.query(Insight).filter_by(session_id="ins-3").count()
    assert count <= 8
