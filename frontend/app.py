import streamlit as st
import requests

st.title("Local LLM Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def estimate_tokens(messages):
    """Simple token estimation: ~4 characters per token"""
    total_chars = 0
    for item in messages:
        total_chars += len(item[1]) + len(item[0]) + 10  # +10 for formatting overhead
    return total_chars // 4  # Rough estimation

def check_token_limit(messages, max_tokens=3000):
    """Check if estimated tokens exceed limit"""
    estimated_tokens = estimate_tokens(messages)
    return estimated_tokens, estimated_tokens > max_tokens

# Fetch provider info
provider = "unknown"
try:
    resp = requests.get("http://localhost:8000/health")
    if resp.status_code == 200:
        provider = resp.json().get("model_provider", "unknown")
except Exception:
    pass

st.caption(f"**Current Model Provider:** `{provider}`")

# Display historical messages
for item in st.session_state["messages"]:
    role = item[0]
    msg = item[1]
    thoughts = item[2] if len(item) > 2 else []
    display_role = "assistant" if role == "bot" else role
    with st.chat_message(display_role):
        st.markdown(msg, unsafe_allow_html=True)
        if thoughts:
            with st.expander("Agent Thought Process"):
                for step in thoughts:
                    st.write(f"- {step}")

# Chat input
if user_input := st.chat_input("Message the bot..."):
    temp_messages = st.session_state["messages"] + [("user", user_input)]
    estimated_tokens, exceeds_limit = check_token_limit(temp_messages)
    
    if exceeds_limit:
        st.error(f"⚠️ Conversation too long! Estimated tokens: {estimated_tokens}/3000. Please start a new chat or clear the conversation.")
        if st.button("Clear Chat"):
            st.session_state["messages"] = []
            st.rerun()
    else:
        st.session_state["messages"].append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    formatted_messages = [{"role": "assistant" if item[0] == "bot" else item[0], "content": item[1]} 
                                        for item in st.session_state["messages"]]
                    
                    resp = requests.post("http://localhost:8000/chat", json={"messages": formatted_messages})
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("response", "[No response generated]")
                        finish_reason = data.get("finish_reason", "unknown")
                        intermediate_steps = data.get("intermediate_steps", [])
                        
                        if finish_reason == "length":
                            answer += "\n\n*(⚠️ **Note:** cut off because token limit.)*"
                    else:
                        answer = f"Error: {resp.text}"
                        intermediate_steps = []
                except Exception as e:
                    answer = f"Error: {e}"
                    intermediate_steps = []
                
                st.markdown(answer, unsafe_allow_html=True)
                if intermediate_steps:
                    with st.expander("Agent Thought Process"):
                        for step in intermediate_steps:
                            st.write(f"- {step}")
                            
                st.session_state["messages"].append(("assistant", answer, intermediate_steps))
