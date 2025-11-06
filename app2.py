#frontend file
import streamlit as st
from api_client import upload_file, chat_query
from PIL import Image
import time

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AstraMind",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------
# HEADER
# ----------------------------
st.markdown("<h1 style='text-align:center; color:#5B5FE9;'>AstraMind</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>AI-Powered Knowledge System</p>", unsafe_allow_html=True)

try:
    st.image("assets/logo.png", width=200)
except:
    pass

st.divider()

# ----------------------------
# FILE UPLOAD SECTION
# ----------------------------
st.subheader("📁 Upload Files")

file = st.file_uploader(
    "Choose a file (PDF, DOCX, Image, or Batch ):",
    type=["pdf", "docx", "png", "jpg", "jpeg", ]
)

upload_type = st.selectbox(
    "Select Upload Type",
    ("pdf", "docx", "image", "batch"),
    index=0
)

if st.button("Upload"):
    if file is not None:
        with st.spinner(f"Uploading {file.name}..."):
            response = upload_file(file, upload_type)
            if "error" in response:
                st.error(response["error"])
            else:
                st.success(f"{upload_type.upper()} uploaded successfully!")
                st.json(response)
    else:
        st.warning("Please select a file first.")

st.divider()

# ----------------------------
# CHAT SECTION
# ----------------------------
st.subheader("💬 Chat with AstraMind")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Chat input box
if prompt := st.chat_input("Ask AstraMind..."):
    # Append user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Query backend
    with st.spinner("Thinking..."):
        response = chat_query(prompt)
        answer = response.get("answer", "⚠️ No response received.")
        context_used = response.get("context_used", [])

    # Append assistant message
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)

    # Show retrieved context (optional)
    if context_used:
        with st.expander("🔍 Context used by AstraMind"):
            for c in context_used:
                st.markdown(f"- {c.strip()}")

# ----------------------------
# FOOTER
# ----------------------------
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ by <b>Sachin</b></p>",
    unsafe_allow_html=True,
)

