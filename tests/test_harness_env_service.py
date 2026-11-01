"""Environment service: session lifecycle, retry safety and scoring order."""

import pytest
from fastapi.testclient import TestClient

from mobilerun.harness.env_service import scoring
from mobilerun.harness.env_service.app import create_env_app
from mobilerun.harness.protocol import ErrorCode, ProtocolError
from mobilerun.harness.testing import ScriptedBackend


def make_client(backend: ScriptedBackend, **kwargs) -> TestClient:
    return TestClient(create_env_app([backend], **kwargs))


def _tap(index: int, **extra):
    return {
        "action": {"op": "tap", "args": {"x": 12, "y": 34}},
        "step_index": index,
        **extra,
    }


def test_health_and_task_catalogue():
    backend = ScriptedBackend(tasks=["Alpha", "Beta"])
    with make_client(backend) as client:
        health = client.get("/v1/health").json()
        assert health["ok"] is True
        assert health["protocol_version"] == "1.0"
        assert health["backends"][0]["id"] == "fake"

        tasks = client.get("/v1/tasks").json()
        assert [task["id"] for task in tasks] == ["Alpha", "Beta"]
        assert tasks[0]["instruction"] == "do Alpha"


def test_session_lifecycle_end_to_end():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        created = client.post("/v1/sessions", json={"task_id": "Alpha"})
        assert created.status_code == 200
        session = created.json()
        session_id = session["session_id"]
        assert session["task"]["id"] == "Alpha#0"
        assert session["observation"]["seq"] == 1
        assert session["capabilities"]["has_ui_tree"] is True

        stepped = client.post(f"/v1/sessions/{session_id}/step", json=_tap(1))
        assert stepped.status_code == 200
        assert stepped.json()["ok"] is True
        assert backend.actions[-1].op == "tap"

        observed = client.post(f"/v1/sessions/{session_id}/observe", json={})
        assert observed.json()["seq"] == 2

        finished = client.post(
            f"/v1/sessions/{session_id}/finish",
            json={"declared_done": True, "answer": "42"},
        )
        assert finished.status_code == 200
        assert finished.json()["score"] == 1.0
        assert finished.json()["gated_score"] == 1.0
        assert backend.answers == ["42"]

        closed = client.delete(f"/v1/sessions/{session_id}")
        assert closed.json()["ok"] is True
        assert backend.teardowns == ["Alpha#0"]


def test_replaying_a_step_never_touches_the_device_twice():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        first = client.post(f"/v1/sessions/{session_id}/step", json=_tap(1)).json()
        replay = client.post(f"/v1/sessions/{session_id}/step", json=_tap(1)).json()

        assert replay == first
        assert len(backend.actions) == 1


def test_out_of_order_step_index_is_rejected():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        response = client.post(f"/v1/sessions/{session_id}/step", json=_tap(7))
        assert response.status_code == 409
        assert response.json()["code"] == "session_state"


def test_stale_observation_is_rejected_with_the_current_frame():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        response = client.post(
            f"/v1/sessions/{session_id}/step", json=_tap(1, expect_seq=99)
        )
        assert response.status_code == 409
        body = response.json()
        assert body["code"] == "stale_observation"
        assert body["detail"]["expected_seq"] == 1
        assert body["detail"]["observation"]["seq"] == 1
        assert backend.actions == []


def test_finish_is_idempotent_and_closes_the_session():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        first = client.post(
            f"/v1/sessions/{session_id}/finish", json={"declared_done": True}
        ).json()
        second = client.post(
            f"/v1/sessions/{session_id}/finish", json={"declared_done": False}
        ).json()

        assert second == first
        assert backend.score_reads == 1

        late = client.post(f"/v1/sessions/{session_id}/step", json=_tap(1))
        assert late.status_code == 409


def test_undeclared_episode_is_gated_to_zero():
    backend = ScriptedBackend(tasks=["Alpha"], score=1.0)
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        finished = client.post(
            f"/v1/sessions/{session_id}/finish", json={"declared_done": False}
        ).json()
        assert finished["score"] == 1.0
        assert finished["gated_score"] == 0.0


def test_zero_score_is_rechecked_once(monkeypatch):
    monkeypatch.setattr(scoring, "_SCORE_RECHECK_DELAY_S", 0.0)
    scores = iter([0.0, 1.0])

    def score_fn(_backend):
        return next(scores, 0.0)

    backend = ScriptedBackend(tasks=["Alpha"], score_fn=score_fn)
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        finished = client.post(
            f"/v1/sessions/{session_id}/finish", json={"declared_done": True}
        ).json()
        assert backend.score_reads == 2
        assert finished["score"] == 1.0
        assert finished["gated_score"] == 1.0


def test_no_recheck_when_the_first_read_succeeds():
    backend = ScriptedBackend(tasks=["Alpha"], score=1.0)
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        client.post(f"/v1/sessions/{session_id}/finish", json={"declared_done": True})
        assert backend.score_reads == 1


def test_a_busy_backend_reports_capacity_instead_of_blocking():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        assert client.post("/v1/sessions", json={"task_id": "Alpha"}).status_code == 200
        response = client.post("/v1/sessions", json={"task_id": "Alpha"})
        assert response.status_code == 503
        assert response.headers["retry-after"]
        assert response.json()["code"] == "capacity"


def test_unknown_session_and_task_are_404():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        assert client.get("/v1/sessions/nope").status_code == 404
        assert (
            client.post("/v1/sessions", json={"task_id": "Missing"}).status_code == 404
        )


def test_a_failing_device_action_is_feedback_not_a_failed_episode():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        response = client.post(
            f"/v1/sessions/{session_id}/step",
            json={
                "action": {"op": "open_app", "args": {"app": "explode"}},
                "step_index": 1,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert "exploded" in body["error"]
        assert client.get(f"/v1/sessions/{session_id}").json()["state"] == "running"


def test_include_observation_returns_the_post_action_frame():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        body = client.post(
            f"/v1/sessions/{session_id}/step",
            json={**_tap(1), "include_observation": True},
        ).json()
        assert body["observation"]["seq"] == 2


def test_disallowed_action_shape_is_rejected_before_the_device():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        response = client.post(
            f"/v1/sessions/{session_id}/step",
            json={"action": {"op": "teleport", "args": {}}, "step_index": 1},
        )
        assert response.status_code == 422
        assert backend.actions == []


@pytest.mark.parametrize("include_system", [True, False])
def test_device_apps_and_date(include_system):
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        apps = client.get(
            f"/v1/sessions/{session_id}/device/apps",
            params={"include_system": include_system},
        ).json()
        assert apps["apps"][0]["package"] == "com.example"
        assert apps["packages"] == ["com.example"]
        date = client.get(f"/v1/sessions/{session_id}/device/date").json()
        assert date["date"] == "2026-01-01T00:00:00"


def test_bearer_token_is_enforced_when_configured():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend, token="s3cret") as client:
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/tasks").status_code == 401
        authorized = client.get("/v1/tasks", headers={"authorization": "Bearer s3cret"})
        assert authorized.status_code == 200


class _FlakyObserve(ScriptedBackend):
    """A backend whose device link drops after the session is opened."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_observations = False

    async def observe(self, *, seq: int):
        if self.fail_observations:
            raise ProtocolError(
                503, ErrorCode.BACKEND_UNAVAILABLE, "device disconnected"
            )
        return await super().observe(seq=seq)


def test_step_survives_a_failed_post_action_observation():
    backend = _FlakyObserve(tasks=["Alpha"])
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        backend.fail_observations = True
        response = client.post(
            f"/v1/sessions/{session_id}/step",
            json={
                "action": {"op": "tap", "args": {"x": 1, "y": 2}},
                "step_index": 1,
                "include_observation": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    # The action ran; only the follow-up frame was unavailable.
    assert body["ok"] is True
    assert body["observation"] is None


def test_observe_maps_a_disconnect_to_503():
    backend = _FlakyObserve(tasks=["Alpha"])
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        backend.fail_observations = True
        response = client.post(f"/v1/sessions/{session_id}/observe?fresh=true")

    assert response.status_code == 503
    assert response.json()["code"] == "backend_unavailable"


def test_stabilizes_before_each_readable_frame():
    backend = ScriptedBackend(tasks=["Alpha"])
    with make_client(backend) as client:
        session_id = client.post("/v1/sessions", json={"task_id": "Alpha"}).json()[
            "session_id"
        ]
        # reset() leaves the UI mid-transition: the first frame waits for it.
        assert backend.stabilize_count == 1

        client.post(
            f"/v1/sessions/{session_id}/step",
            json={
                "action": {"op": "tap", "args": {"x": 1, "y": 2}},
                "step_index": 1,
                "include_observation": True,
            },
        )
        # The post-action frame must be taken after the UI settled.
        assert backend.stabilize_count == 2

        client.post(
            f"/v1/sessions/{session_id}/step",
            json={
                "action": {"op": "tap", "args": {"x": 1, "y": 2}},
                "step_index": 2,
                "include_observation": False,
            },
        )
        assert backend.stabilize_count == 2
