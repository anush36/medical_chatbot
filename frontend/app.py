# Copyright (c) 2026 Anush Agarwal. All rights reserved.
# This code is proprietary and provided for public review and educational purposes.
# Unauthorized use, reproduction, or distribution is strictly prohibited.

import streamlit as st
import requests
import base64
import io

st.title("Vesa Health Assistant")
st.caption("A verified and safe assistant for answering everyday medical questions.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def estimate_tokens(messages):
    """Simple token estimation: ~4 characters per token"""
    total_chars = 0
    for item in messages:
        content = item[1]
        if isinstance(content, list):
            # For multimodal, estimate based on text parts
            for part in content:
                if part.get("type") == "text":
                    total_chars += len(part.get("text", ""))
        else:
            total_chars += len(content)
        total_chars += 10  # +10 for formatting overhead
    return total_chars // 4

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

# File Uploader helper
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Display historical messages
for item in st.session_state["messages"]:
    role = item[0]
    content = item[1]
    thoughts = item[2] if len(item) > 2 else []
    display_role = "assistant" if role == "bot" else role
    with st.chat_message(display_role):
        if isinstance(content, list):
            for part in content:
                if part["type"] == "text":
                    st.markdown(part["text"], unsafe_allow_html=True)
                elif part["type"] == "image_url":
                    st.image(part["image_url"]["url"])
        else:
            st.markdown(content, unsafe_allow_html=True)
            
        if thoughts:
            with st.expander("Agent Thought Process"):
                for step in thoughts:
                    st.write(f"- {step}")

# Chat input with integrated file uploader
if prompt := st.chat_input("Message the bot...", accept_file=True, file_type=["jpg", "jpeg", "png", "pdf"]):
    # prompt is a dictionary when accept_file=True and a file/text is submitted
    user_input = prompt.get("text", "")
    uploaded_files = prompt.get("files", [])
    uploaded_file = uploaded_files[0] if uploaded_files else None
    
    current_content = user_input
    
    # Handle File Upload logic
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                resp = requests.post("http://localhost:8000/parse-pdf", files=files)
                if resp.status_code == 200:
                    pdf_text = resp.json().get("text", "")
                    # Combine PDF text context with the user's question. If no text, just send the context.
                    question = user_input if user_input else "Please analyze this document."
                    current_content = f"CONTEXT FROM PDF:\n{pdf_text}\n\nUSER QUESTION: {question}"
                else:
                    st.error("Failed to parse PDF.")
        elif uploaded_file.type in ["image/jpeg", "image/png"]:
            base64_image = encode_image(uploaded_file)
            current_content = []
            if user_input:
                current_content.append({"type": "text", "text": user_input})
            else:
                current_content.append({"type": "text", "text": "Please analyze this image."})
                
            current_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{uploaded_file.type};base64,{base64_image}"}
            })

    temp_messages = st.session_state["messages"] + [("user", current_content)]
    estimated_tokens, exceeds_limit = check_token_limit(temp_messages)
    
    if exceeds_limit:
        st.error(f"⚠️ Conversation too long! Estimated tokens: {estimated_tokens}/3000. Please start a new chat or clear the conversation.")
        if st.button("Clear Chat"):
            st.session_state["messages"] = []
            st.rerun()
    else:
        st.session_state["messages"].append(("user", current_content))
        with st.chat_message("user"):
            if isinstance(current_content, list):
                if user_input:
                    st.markdown(user_input)
                if uploaded_file:
                    st.image(uploaded_file)
            else:
                # If it's a pdf, maybe don't display the full extracted payload to the user but just what they typed
                display_text = user_input if uploaded_file and uploaded_file.type == "application/pdf" else current_content
                st.markdown(display_text)
                if uploaded_file and uploaded_file.type == "application/pdf":
                    st.caption(f"📎 Attached PDF: {uploaded_file.name}")
            
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
