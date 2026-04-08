from fastapi.testclient import TestClient

from customer_support_env.server import app


client = TestClient(app)


def test_reset_allows_empty_body():
    response = client.post("/reset")
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] in {"easy_login_triage", "medium_refund_reply", "hard_queue_planning"}


def test_state_get_returns_current_state():
    client.post("/reset")
    response = client.get("/state")
    assert response.status_code == 200
    payload = response.json()
    assert payload["episode_id"]


def test_state_post_allows_empty_body():
    client.post("/reset")
    response = client.post("/state")
    assert response.status_code == 200


def test_step_post_allows_empty_body():
    client.post("/reset")
    response = client.post("/step")
    assert response.status_code == 200
    payload = response.json()
    assert "observation" in payload
