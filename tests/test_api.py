import importlib
import json

import pytest
from fastapi.testclient import TestClient

from windfc.config import DEFAULT_CONFIG


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The API on an empty data folder, via a copy of config.yaml pointing at tmp_path."""
    config = tmp_path / "config.yaml"
    text = DEFAULT_CONFIG.read_text().replace("data_dir: data", f"data_dir: {tmp_path.as_posix()}/data")
    config.write_text(text)
    monkeypatch.setenv("WINDFC_CONFIG", str(config))
    import windfc.api
    api = importlib.reload(windfc.api)
    yield TestClient(api.app), api.store
    monkeypatch.delenv("WINDFC_CONFIG")
    importlib.reload(windfc.api)


def test_evaluation_missing_is_404(client):
    http, _ = client
    assert http.get("/evaluation").status_code == 404
    assert http.get("/evaluation/chart").status_code == 404
    assert "No verification yet" in http.get("/").text


def test_evaluation_json_and_chart(client, evaluation):
    http, store = client
    store.evaluation_path().parent.mkdir(parents=True)
    store.evaluation_path().write_text(json.dumps(evaluation))
    assert http.get("/evaluation").json()["runs_scored"] == evaluation["runs_scored"]
    chart = http.get("/evaluation/chart")
    assert chart.headers["content-type"] == "image/png"
    assert chart.content.startswith(b"\x89PNG")
    assert http.get("/evaluation/rank-histogram").status_code == 404  # not written by this test
    assert '<img src="/evaluation/chart"' in http.get("/").text
