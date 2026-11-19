"""Central settings. Every value can be overridden with a DWAGENT_* environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECRET = "local-demo-secret-change-me"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"DWAGENT_{name}", default)


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    kb_dir: Path = ROOT / "kb"

    # LLM. The default is a local model served by Ollama: nothing leaves the machine.
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://127.0.0.1:11434"))
    model: str = field(default_factory=lambda: _env("MODEL", "qwen3:14b"))
    num_ctx: int = field(default_factory=lambda: int(_env("NUM_CTX", "8192")))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "180")))

    # Earlier RAG project (deepwise_rag) running as a sidecar HTTP service.
    rag_url: str = field(default_factory=lambda: _env("RAG_URL", "http://127.0.0.1:8100"))
    rag_api_key: str = field(default_factory=lambda: _env("RAG_API_KEY", "dev-research-key"))

    # Deployment profile. "onprem" blocks every outbound call that is not to this host.
    profile: str = field(default_factory=lambda: _env("PROFILE", "onprem"))

    # Web session signing key. The default is for the local demo only.
    secret_key: str = field(default_factory=lambda: _env("SECRET_KEY", DEFAULT_SECRET))

    # Public mode: the portal is reachable from the internet through a tunnel. It refuses the
    # default secret, takes account passwords from a local file instead of the source code,
    # never sends passwords to the browser, and applies tighter request limits.
    public: bool = field(default_factory=lambda: _env("PUBLIC", "0").lower() in ("1", "true", "yes"))
    credentials_file: str = field(default_factory=lambda: _env("CREDENTIALS_FILE", ""))

    @property
    def lidc_dir(self) -> Path:
        return self.data_dir / "lidc"

    @property
    def runtime_dir(self) -> Path:
        p = self.data_dir / "runtime"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.runtime_dir / "dwagent.sqlite"


def get_settings() -> Settings:
    return Settings()


class EgressBlocked(RuntimeError):
    pass


def assert_egress_allowed(url: str) -> None:
    """In the on-prem profile every outbound call must stay on this host or the hospital LAN
    names configured in DWAGENT_ALLOWED_HOSTS. Patient data never leaves the site."""
    from urllib.parse import urlparse
    if get_settings().profile != "onprem":
        return
    host = (urlparse(url).hostname or "").lower()
    allowed = {"127.0.0.1", "localhost", "::1"} | {h.strip().lower() for h in _env("ALLOWED_HOSTS", "").split(",") if h.strip()}
    if host not in allowed:
        raise EgressBlocked(f"on-prem profile: outbound call to {host!r} is not allowed")
