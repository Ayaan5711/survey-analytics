from __future__ import annotations
from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    row_count = Column(Integer)
    profile_path = Column(String)
    data_path = Column(String)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    pinned_charts = relationship("PinnedChart", back_populates="session", cascade="all, delete-orphan")
    insights = relationship("Insight", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    generated_code = Column(Text, nullable=True)
    chart_paths = Column(JSON, nullable=True)
    follow_ups = Column(JSON, nullable=True)
    caveats = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="messages")


class PinnedChart(Base):
    __tablename__ = "pinned_charts"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    chart_path = Column(String, nullable=False)
    title = Column(String, nullable=True)
    source_message_id = Column(String, ForeignKey("chat_messages.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="pinned_charts")


class Insight(Base):
    __tablename__ = "insights"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    rank = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    supporting_tool_calls = Column(JSON, nullable=True)
    chart_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="insights")


class Comparison(Base):
    __tablename__ = "comparisons"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    base_session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    compare_session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    diff_summary = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
