import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    """读取项目根目录的 .env。已存在的环境变量优先。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_api_base: str
    deepseek_model: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    root: Path
    entity_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
            deepseek_api_base=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1").strip(),
            deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat",
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687").strip(),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j").strip() or "neo4j",
            neo4j_password=os.environ.get("NEO4J_PASSWORD", "").strip(),
            neo4j_database=os.environ.get("NEO4J_DATABASE", "neo4j").strip() or "neo4j",
            root=ROOT,
            entity_dir=ROOT / "data" / "ent_aug",
        )


settings = Settings.from_env()
