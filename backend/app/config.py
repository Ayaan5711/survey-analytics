from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    azure_openai_endpoint: str = field(default_factory=lambda: os.environ["AZURE_OPENAI_ENDPOINT"])
    azure_openai_api_key: str = field(default_factory=lambda: os.environ["AZURE_OPENAI_API_KEY"])
    azure_openai_api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"))
    deployment_primary: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_PRIMARY", "gpt-4-1"))
    deployment_fallback: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_FALLBACK", "gpt-4o-mini"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data/sessions")))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./survey_analytics.db"))


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
