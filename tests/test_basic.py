import requests

def test_health_endpoint():
    """Test the health check endpoint."""
    url = "http://localhost:8000/health"
    resp = requests.get(url)
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_provider" in data
    assert "model_available" in data
    assert data["status"] == "healthy"

def test_chat_endpoint():
    """Test the chat endpoint."""
    url = "http://localhost:8000/chat"
    data = {"prompt": "Hello!"}
    resp = requests.post(url, json=data)
    assert resp.status_code == 200
    assert "response" in resp.json()
