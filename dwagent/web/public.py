"""Start the portal in public mode, for exposure through a tunnel.

    python -m dwagent.web.public            start on 127.0.0.1:8200 (DWAGENT_PORT to change)
    python -m dwagent.web.public --rotate   new account passwords and session secret, then start
    python -m dwagent.web.public --setup    create the secret and passwords and exit

The first start writes two files under data/runtime/ (never committed):
    public_secret.key          random session-signing secret
    public_credentials.json    one random password per demo account

The process listens on this machine only. A tunnel program on the same machine forwards the
public HTTPS address to it; the model, the retrieval service and the DICOM gateway are not
exposed. Forwarded-address headers are trusted from 127.0.0.1 only, which is the tunnel.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path

from ..config import get_settings
from ..profiles import DEMO_USERS

_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"      # no look-alike characters: typed from a slide


def new_password() -> str:
    return "-".join("".join(secrets.choice(_ALPHABET) for _ in range(4)) for _ in range(4))


def setup(runtime: Path, rotate: bool = False) -> tuple[Path, Path]:
    secret_file, cred_file = runtime / "public_secret.key", runtime / "public_credentials.json"
    if rotate or not secret_file.exists():
        secret_file.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    creds = {} if rotate or not cred_file.exists() else json.loads(cred_file.read_text(encoding="utf-8"))
    users = [u for u, *_ in DEMO_USERS]
    if any(u not in creds for u in users):
        creds = {u: creds.get(u) or new_password() for u in users}
        cred_file.write_text(json.dumps(creds, indent=2), encoding="utf-8")
    return secret_file, cred_file


def main(argv: list[str]) -> None:
    secret_file, cred_file = setup(get_settings().runtime_dir, rotate="--rotate" in argv)
    print(f"account passwords: {cred_file}")
    if "--setup" in argv:
        return
    os.environ["DWAGENT_PUBLIC"] = "1"
    os.environ["DWAGENT_SECRET_KEY"] = secret_file.read_text(encoding="utf-8").strip()
    os.environ["DWAGENT_CREDENTIALS_FILE"] = str(cred_file)
    port = int(os.environ.get("DWAGENT_PORT", "8200"))
    import uvicorn
    print(f"public mode, listening on http://127.0.0.1:{port} (this machine only)")
    uvicorn.run("dwagent.web.app:app", host="127.0.0.1", port=port, proxy_headers=True, forwarded_allow_ips="127.0.0.1")


if __name__ == "__main__":
    main(sys.argv[1:])
