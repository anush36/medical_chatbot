import streamlit as st
import requests

st.title("Local LLM Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("You:", "", key="input")

if st.button("Send") and user_input:
    st.session_state["messages"].append(("user", user_input))
    with st.spinner("Thinking..."):
        try:
            resp = requests.post("http://localhost:8000/chat", json={"prompt": user_input})
            if resp.status_code == 200:
                answer = resp.json()["response"]
            else:
                answer = f"Error: {resp.text}"
        except Exception as e:
            answer = f"Error: {e}"
        st.session_state["messages"].append(("bot", answer))

for role, msg in st.session_state["messages"]:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")
