import uuid

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from chatbot import get_chatbot

# --------- Initialize Chatbot ---------
chatbot = get_chatbot()

# --------- Initialize Session State ---------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}  # {session_id: {"messages": [], "title": str}}
if "active_session" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.active_session = new_id
    st.session_state.conversations[new_id] = {"messages": [], "title": "New Chat"}

# --------- Title Generator LLM ---------
llm = ChatOllama(model="llama3", temperature=0)
title_prompt = ChatPromptTemplate.from_template(
    "Generate a short (max 6 words) title summarizing this user query:\n\n{query}"
)

def generate_title(query: str) -> str:
    response = llm.invoke(title_prompt.format_messages(query=query))
    return response.content.strip()

# --------- UI ---------
st.title("🤖 Conversational RAG Chatbot")

session_data = st.session_state.conversations[st.session_state.active_session]

# Display past messages
for message in session_data["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask me anything about the PDFs..."):
    session_data["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = chatbot.ask(prompt, session_id=st.session_state.active_session)
    session_data["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    # Update session title if it's still "New Chat"
    if session_data["title"] == "New Chat":
        session_data["title"] = generate_title(session_data["messages"])

# --------- Sidebar ---------
with st.sidebar:
    st.subheader("💬 Conversations")

    if st.button("➕ Start New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.conversations[new_id] = {"messages": [], "title": "New Chat"}
        st.session_state.active_session = new_id
        st.rerun()

    # List sessions with clickable titles
    for sid, convo in st.session_state.conversations.items():
        if st.button(convo["title"], key=sid):
            st.session_state.active_session = sid
            st.rerun()