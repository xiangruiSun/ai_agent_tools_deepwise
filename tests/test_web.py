"""Tests of the portal's access model and of public mode. No model is called and no CT data is
needed: every request here is answered before an agent or an image is touched."""
from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient

from dwagent import config, store
from dwagent.profiles import DEMO_USERS
from dwagent.web import public as public_mode

LOCAL_PW = {u: pw for u, pw, *_ in DEMO_USERS}


def load_app(monkeypatch, tmp_path, **env):
    """Import the portal fresh with its own data directory and environment."""
    for k in ("PUBLIC", "SECRET_KEY", "CREDENTIALS_FILE"):
        monkeypatch.delenv(f"DWAGENT_{k}", raising=False)
    monkeypatch.setenv("DWAGENT_DATA_DIR", str(tmp_path))
    for k, v in env.items():
        monkeypatch.setenv(f"DWAGENT_{k}", v)
    monkeypatch.setattr(store, "_default", None)
    from dwagent.web import app as module
    return importlib.reload(module)


def sign_in(client, user, password):
    return client.post("/api/login", json={"username": user, "password": password})


@pytest.fixture
def local(monkeypatch, tmp_path):
    module = load_app(monkeypatch, tmp_path)
    yield module, TestClient(module.app)
    monkeypatch.setattr(store, "_default", None)


@pytest.fixture
def public(monkeypatch, tmp_path):
    secret_file, cred_file = public_mode.setup(tmp_path)
    module = load_app(monkeypatch, tmp_path, PUBLIC="1", SECRET_KEY=secret_file.read_text(encoding="utf-8"),
                      CREDENTIALS_FILE=str(cred_file))
    yield module, TestClient(module.app, base_url="https://testserver"), json.loads(cred_file.read_text(encoding="utf-8"))
    monkeypatch.setattr(store, "_default", None)


# ---------------------------------------------------------------- access model
def test_routes_need_a_session(local):
    _, c = local
    for path in ("/api/me", "/api/worklist", "/api/ops/overview", "/api/my/report", "/api/memory"):
        assert c.get(path).status_code == 401


@pytest.mark.parametrize("user,allowed,denied", [
    ("patient.0005", ["/api/my/report"], ["/api/worklist", "/api/approvals", "/api/ops/audit", "/api/studies/LIDC-IDRI-0212"]),
    ("dr.chen", ["/api/approvals"], ["/api/my/report", "/api/ops/audit", "/api/ops/runs"]),
    ("eng.li", ["/api/ops/audit", "/api/ops/runs"], ["/api/my/report", "/api/worklist", "/api/studies/LIDC-IDRI-0005"]),
])
def test_role_matrix(local, user, allowed, denied):
    _, c = local
    assert sign_in(c, user, LOCAL_PW[user]).status_code == 200
    for path in allowed:
        assert c.get(path).status_code == 200, path
    for path in denied:
        assert c.get(path).status_code == 403, path


def test_patient_cannot_view_another_patients_image(local):
    _, c = local
    sign_in(c, "patient.0005", LOCAL_PW["patient.0005"])
    assert c.get("/api/studies/LIDC-IDRI-0212/slice/10.png").status_code == 403


def test_role_is_not_taken_from_the_request(local):
    _, c = local
    r = c.post("/api/login", json={"username": "patient.0005", "password": LOCAL_PW["patient.0005"], "role": "doctor"})
    assert r.json()["role"] == "patient"
    assert c.get("/api/worklist").status_code == 403


def test_tampered_session_is_rejected(local):
    _, c = local
    sign_in(c, "dr.chen", LOCAL_PW["dr.chen"])
    token = c.cookies.get("dw_session")
    c.cookies.clear()
    assert c.get("/api/me", headers={"Cookie": f"dw_session={token}"}).status_code == 200
    forged = token[:-6] + ("AAAAAA" if not token.endswith("AAAAAA") else "BBBBBB")
    assert c.get("/api/me", headers={"Cookie": f"dw_session={forged}"}).status_code == 401


def test_login_throttle(local):
    _, c = local
    codes = [sign_in(c, "dr.chen", "wrong-password").status_code for _ in range(9)]
    assert codes[:8] == [401] * 8 and codes[8] == 429
    assert sign_in(c, "dr.chen", LOCAL_PW["dr.chen"]).status_code == 429      # the right password does not skip the wait


def test_security_headers(local):
    _, c = local
    h = c.get("/").headers
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
    assert "script-src" not in h["content-security-policy"] and "default-src 'self'" in h["content-security-policy"]
    assert c.get("/api/demo-accounts").headers["cache-control"] == "no-store"


# ---------------------------------------------------------------- admission control
def test_job_limits(local):
    module, c = local
    sign_in(c, "dr.chen", LOCAL_PW["dr.chen"])
    who = module.principal("dr.chen", "doctor", None, "Dr. Chen")

    class Req:
        method, client = "POST", type("C", (), {"host": "203.0.113.7"})()
        url = type("U", (), {"path": "/api/chat"})()

    key = module.admit(who, Req)
    for i in range(module.MAX_RUNNING_PER_CLIENT):
        module.JOBS[f"r{i}"] = {"owner": "dr.chen", "client": key, "status": "running", "result": None, "at": 0}
    with pytest.raises(module.HTTPException) as exc:
        module.admit(who, Req)
    assert exc.value.status_code == 429 and "current request" in exc.value.detail

    other = type("R", (), {"method": "POST", "client": type("C", (), {"host": "198.51.100.9"})(), "url": Req.url})
    assert module.admit(who, other)                      # another visitor on the same account is not blocked

    for j in module.JOBS.values():
        j["status"] = "done"
    module._job_starts[key] = [module.time.time()] * module.MAX_JOBS_PER_10_MIN
    with pytest.raises(module.HTTPException) as exc:
        module.admit(who, Req)
    assert "limit" in exc.value.detail
    assert not module.JOBS                               # finished jobs older than an hour were pruned
    module._job_starts.clear()


# ---------------------------------------------------------------- public mode
def test_public_refuses_default_secret(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        load_app(monkeypatch, tmp_path, PUBLIC="1")
    load_app(monkeypatch, tmp_path)


def test_public_refuses_source_passwords(monkeypatch, tmp_path):
    weak = tmp_path / "creds.json"
    weak.write_text(json.dumps(LOCAL_PW), encoding="utf-8")
    with pytest.raises(RuntimeError, match="own password"):
        load_app(monkeypatch, tmp_path, PUBLIC="1", SECRET_KEY="x" * 40, CREDENTIALS_FILE=str(weak))
    load_app(monkeypatch, tmp_path)


def test_public_hides_passwords_and_docs(public):
    _, c, creds = public
    accounts = c.get("/api/demo-accounts").json()
    assert len(accounts) == len(DEMO_USERS) and all(a["password"] is None for a in accounts)
    body = json.dumps(accounts) + c.get("/static/app.js").text
    assert not any(pw in body for pw in creds.values())
    assert c.get("/api/docs").status_code == 404 and c.get("/api/openapi.json").status_code == 404


def test_public_login_uses_generated_passwords(public):
    _, c, creds = public
    assert sign_in(c, "dr.chen", LOCAL_PW["dr.chen"]).status_code == 401      # the password in the source no longer works
    r = sign_in(c, "dr.chen", creds["dr.chen"])
    assert r.status_code == 200 and r.json()["role"] == "doctor"
    cookie = r.headers["set-cookie"].lower()
    assert "secure" in cookie and "httponly" in cookie and "samesite=lax" in cookie
    assert "max-age=31536000" in r.headers["strict-transport-security"] and "noindex" in r.headers["x-robots-tag"]


def test_generated_passwords_are_distinct_and_stable(tmp_path):
    _, cred_file = public_mode.setup(tmp_path)
    first = json.loads(cred_file.read_text(encoding="utf-8"))
    assert len(set(first.values())) == len(DEMO_USERS) and all(len(p) >= 19 for p in first.values())
    assert json.loads(public_mode.setup(tmp_path)[1].read_text(encoding="utf-8")) == first
    assert json.loads(public_mode.setup(tmp_path, rotate=True)[1].read_text(encoding="utf-8")) != first
