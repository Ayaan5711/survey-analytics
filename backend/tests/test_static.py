from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def static_client():
    with TestClient(app) as c:
        yield c


def test_root_returns_html(static_client):
    r = static_client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_app_js_served(static_client):
    r = static_client.get("/app.js")
    assert r.status_code == 200


def test_styles_css_served(static_client):
    r = static_client.get("/styles.css")
    assert r.status_code == 200
