import streamlit as st
from src.services.ai_service import generate_response, get_models_list
import uuid


# Helper function to create a new chat
def create_new_chat():
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[new_chat_id] = []
    st.session_state.current_chat_id = new_chat_id


def build_sidebar():
    with st.sidebar:
        # Button to start a new chat
        if st.button("➕ New Chat", use_container_width=True):
            create_new_chat()
            st.rerun()

        st.divider()

        st.subheader("My Chats")

        # List existing chats as buttons
        for chat_id, messages in st.session_state.chats.items():
            # Create a preview title based on the first user message, or default text
            if messages and len(messages) > 0:
                chat_title = messages[0]["content"][:20] + "..."
            else:
                chat_title = "Empty Chat"

            # Highlight the active chat visually
            is_active = chat_id == st.session_state.current_chat_id
            button_type = "primary" if is_active else "secondary"

            if st.button(
                chat_title, key=chat_id, type=button_type, use_container_width=True
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()


def build_ui():
    st.title("Oracle RAG")

    if "chats" not in st.session_state:
        # Dictionary to hold all chats: { chat_id: [{"role": "user", "content": "..."}] }
        st.session_state.chats = {}

    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
        
    build_sidebar()

    # Create an initial chat if the app is just starting
    if not st.session_state.chats:
        create_new_chat()

    # Get the history for the currently selected chat
    current_chat_history = st.session_state.chats[st.session_state.current_chat_id]

    # Display all previous messages in the active chat
    for message in current_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # st.chat_input stays anchored at the bottom of the screen
    if prompt := st.chat_input("Type your message here..."):

        # 1. Display and store the user's message
        with st.chat_message("user"):
            st.markdown(prompt)
        current_chat_history.append({"role": "user", "content": prompt})

        # 2. Generate and display the assistant's response
        response = generate_response(prompt)

        with st.chat_message("assistant"):
            st.markdown(response.content)
            st.caption(f"Model used to provied the answer: {response.model}.")
        current_chat_history.append({"role": "assistant", "content": response.content})

        # Optionally rerun to update the sidebar title immediately if this was the first message
        if len(current_chat_history) == 2:
            st.rerun()
