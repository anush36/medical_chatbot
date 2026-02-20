import streamlit as st
import requests

st.title("Local LLM Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def estimate_tokens(messages):
    """Simple token estimation: ~4 characters per token"""
    total_chars = 0
    for role, content in messages:
        total_chars += len(content) + len(role) + 10  # +10 for formatting overhead
    return total_chars // 4  # Rough estimation

def check_token_limit(messages, max_tokens=3000):
    """Check if estimated tokens exceed limit"""
    estimated_tokens = estimate_tokens(messages)
    return estimated_tokens, estimated_tokens > max_tokens

user_input = st.text_input("You:", "", key="input")

if st.button("Send") and user_input:
    # Create temporary message list to check token limit
    temp_messages = st.session_state["messages"] + [("user", user_input)]
    estimated_tokens, exceeds_limit = check_token_limit(temp_messages)
    
    if exceeds_limit:
        st.error(f"⚠️ Conversation too long! Estimated tokens: {estimated_tokens}/3000. Please start a new chat or clear the conversation.")
        if st.button("Clear Chat"):
            st.session_state["messages"] = []
            st.rerun()
    else:
        st.session_state["messages"].append(("user", user_input))
        with st.spinner("Thinking..."):
            try:
                # Convert to expected API format
                formatted_messages = [{"role": role if role != "bot" else "assistant", "content": content} 
                                    for role, content in st.session_state["messages"]]
                
                resp = requests.post("http://localhost:8000/chat", json={"messages": formatted_messages})
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("response", "[No response generated]")
                    finish_reason = data.get("finish_reason", "unknown")
                    
                    if finish_reason == "length":
                        answer += "\n\n*(⚠️ **Note:** This response was cut off because it reached the maximum configured token limit. Consider increasing the `MAX_TOKENS` setting in your backend configuration if you need longer answers.)*"
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
