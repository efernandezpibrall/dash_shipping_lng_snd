from app import health, ready, server


def test_health_is_real_json_endpoint():
    routes = {rule.rule for rule in server.url_map.iter_rules()}
    payload = health()

    assert "/health" in routes
    assert payload == {
        "service": "dash_shipping_lng_snd",
        "status": "ok",
    }


def test_ready_reports_database_success(monkeypatch):
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _query):
            return None

    class Engine:
        def connect(self):
            return Connection()

    monkeypatch.setattr(
        "utils.database.get_database_engine",
        lambda: Engine(),
    )

    payload = ready()

    assert payload["status"] == "ready"


def test_ready_reports_database_failure_without_leaking_exception(monkeypatch):
    monkeypatch.setattr(
        "utils.database.get_database_engine",
        lambda: (_ for _ in ()).throw(RuntimeError("secret detail")),
    )

    payload, status_code = ready()

    assert status_code == 503
    assert payload == {
        "service": "dash_shipping_lng_snd",
        "status": "unavailable",
    }
    assert "secret detail" not in str(payload)
