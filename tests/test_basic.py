import requests

def test_chat_endpoint():
    url = "http://localhost:8000/chat"
    data = {"prompt": "Hello!"}
    resp = requests.post(url, json=data)
    assert resp.status_code == 200
    assert "response" in resp.json()
