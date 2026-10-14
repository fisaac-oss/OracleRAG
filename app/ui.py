import random
import uuid
import streamlit as st
from src.services.ai_service import (
    generate_response,
    ingest_rulebook_file,
    get_active_tomes,
)


def inject_arcane_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&display=swap');

        /* Arcane font for headings */
        h1, h2, h3 {
            font-family: 'Cinzel', serif !important;
            letter-spacing: 0.05em;
            color: #E6C280 !important;
            text-shadow: 0 0 12px rgba(212, 175, 55, 0.2);
        }

        /* Buttons with subtle golden glow */
        .stButton button {
            font-family: 'Cinzel', serif !important;
            letter-spacing: 0.03em;
            border-radius: 6px;
            transition: all 0.25s ease;
        }

        .stButton button:hover {
            border-color: #D4AF37 !important;
            box-shadow: 0 0 10px rgba(212, 175, 55, 0.35);
        }

        /* Thematic lore blockquote */
        blockquote {
            border-left: 3px solid #D4AF37 !important;
            background-color: rgba(26, 26, 36, 0.7);
            padding: 12px 18px;
            border-radius: 0 8px 8px 0;
            font-style: italic;
            color: #D6D0C4;
        }

        /* Captions */
        .stCaption {
            font-family: 'Cinzel', serif;
            font-size: 0.78rem;
            color: #A39B8B !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Helper function to create a new consultation
def create_new_chat():
    # If the current active chat is already empty, just keep using it
    current_id = st.session_state.get("current_chat_id")
    if current_id and not st.session_state.chats.get(current_id):
        return

    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[new_chat_id] = []
    st.session_state.current_chat_id = new_chat_id


# In-session utility: Divination Die (d20)
def build_dice_roller():
    if "last_d20_roll" not in st.session_state:
        st.session_state.last_d20_roll = None

    st.subheader("🎲 Divination Die")
    if st.button("Roll d20", use_container_width=True):
        st.session_state.last_d20_roll = random.randint(1, 20)

    if st.session_state.get("last_d20_roll") is not None:
        roll = st.session_state.last_d20_roll
        if roll == 20:
            st.success("**Natural 20!** ✨ Divine favor smiles upon you!")
        elif roll == 1:
            st.error("**Natural 1!** 💀 An ill-fated omen...")
        else:
            st.info(f"The die reveals: **{roll}**")


@st.dialog("📚 The Oracle's Library of Tomes")
def build_upload_dialog():
    st.write("Add your rulebooks, compendiums, or grimoires (`.pdf`, `.docx`, `.md`).")
    uploaded_files = st.file_uploader(
        "Upload rulebooks",
        accept_multiple_files=True,
        type=["pdf", "docx", "md"],
        label_visibility="collapsed",
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.indexed_tomes:
                with st.spinner(f"Inscribing '{file.name}' into sacred archives..."):
                    res = ingest_rulebook_file(file, file.name)
                    if res.success:
                        st.session_state.indexed_tomes.append(file.name)
                        st.success(
                            f"✨ Inscribed **{file.name}** ({res.chunks_count} rule fragments)"
                        )
                    else:
                        st.error(f"Failed to inscribe {file.name}: {res.error_message}")
            else:
                st.info(f"📜 `{file.name}` is already inscribed.")

    if st.session_state.get("indexed_tomes"):
        st.divider()
        st.caption("Active Grimoires in Sanctum:")
        for tome in st.session_state.indexed_tomes:
            st.markdown(f"🕯️ `{tome}`")


def build_sidebar():
    with st.sidebar:
        st.markdown("### 🔮 The Oracle's Sanctum")

        # Action to start a new consultation
        if st.button("✨ Commune Anew", use_container_width=True):
            create_new_chat()
            st.rerun()

        tome_count = len(st.session_state.get("indexed_tomes", []))
        lib_label = f"📖 Library ({tome_count})" if tome_count else "📖 Library"
        if st.button(lib_label, use_container_width=True):
            build_upload_dialog()

        st.divider()

        # Only display recorded visions that actually contain messages
        recorded_chats = [
            (cid, msgs) for cid, msgs in st.session_state.chats.items() if msgs
        ]

        if recorded_chats:
            st.subheader("🕯️ Recorded Visions")
            for chat_id, messages in recorded_chats:
                chat_title = f"📜 {messages[0]['content'][:20]}..."

                # Highlight active vision
                is_active = chat_id == st.session_state.current_chat_id
                button_type = "primary" if is_active else "secondary"

                if st.button(
                    chat_title, key=chat_id, type=button_type, use_container_width=True
                ):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()

            st.divider()

        build_dice_roller()


def build_ui():
    st.set_page_config(
        page_title="The Oracle's Sanctum",
        page_icon="🔮",
        layout="wide",
    )

    inject_arcane_styles()

    # Title & flavor subtitle
    st.title("🔮 The Oracle's Sanctum")
    st.caption("Arbiter of Sacred Rules & Forgotten Tomes")

    # 1. Initialize state BEFORE building sidebar
    if "chats" not in st.session_state:
        st.session_state.chats = {}

    if "indexed_tomes" not in st.session_state:
        st.session_state.indexed_tomes = get_active_tomes()

    if (
        "current_chat_id" not in st.session_state
        or not st.session_state.current_chat_id
    ):
        create_new_chat()

    # 2. Build sidebar after state is guaranteed to exist
    build_sidebar()

    current_chat_history = st.session_state.chats[st.session_state.current_chat_id]

    selected_prompt = None

    # Empty consultation welcome screen with starter questions
    if not current_chat_history:
        st.markdown("""
            > *"Before you lies the vast expanse of rulebooks and compendiums. Speak, traveler, and the Oracle shall divine the true law."*
            """)
        st.write("##### **Sacred Inquiries to Begin:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⚔️ Attacks of Opportunity", use_container_width=True):
                selected_prompt = "How do Attacks of Opportunity work in combat?"
        with col2:
            if st.button("🛡️ Rules of Grappling", use_container_width=True):
                selected_prompt = (
                    "What are the rules and ability checks required for grappling?"
                )
        with col3:
            if st.button("🔥 Dual Spellcasting", use_container_width=True):
                selected_prompt = (
                    "Can a spellcaster cast two leveled spells in a single turn?"
                )

    # Render conversation feed
    for message in current_chat_history:
        avatar = "🎲" if message["role"] == "user" else "🔮"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander(
                    f"📜 Consulted Scriptures ({len(message['citations'])})"
                ):
                    for c in message["citations"]:
                        page_str = f" | Page {c['page']}" if c.get("page") else ""
                        st.markdown(
                            f"**{c.get('source', 'Unknown Tome')}** — *{c.get('section', 'General')}{page_str}*"
                        )
                        st.caption(c.get("text", ""))
            if message.get("model"):
                st.caption(
                    f"🕯️ Channeled through the ethereal essence of: `{message['model']}`"
                )

    # Accept input from sticky bar or prompt pill
    user_input = st.chat_input(
        "Seek counsel from the sacred tomes, traveler...",
    )
    prompt = user_input or selected_prompt

    if prompt:
        # 1. User message
        with st.chat_message("user", avatar="🎲"):
            st.markdown(prompt)
        current_chat_history.append({"role": "user", "content": prompt})

        # 2. Oracle response
        with st.spinner("Consulting the ancient scriptures..."):
            response = generate_response(prompt)

        with st.chat_message("assistant", avatar="🔮"):
            st.markdown(response.content)
            if response.citations:
                with st.expander(
                    f"📜 Consulted Scriptures ({len(response.citations)})"
                ):
                    for c in response.citations:
                        page_str = f" | Page {c.page}" if c.page else ""
                        st.markdown(f"**{c.source}** — *{c.section}{page_str}*")
                        st.caption(c.text)
            model_used = response.model or "Sacred Archive"
            st.caption(f"🕯️ Channeled through the ethereal essence of: `{model_used}`")

        citations_data = [
            c.model_dump() if hasattr(c, "model_dump") else c
            for c in response.citations
        ] if response.citations else []

        current_chat_history.append(
            {
                "role": "assistant",
                "content": response.content,
                "model": response.model,
                "citations": citations_data,
            }
        )

        # Rerun to update sidebar title if it's the first exchange
        if len(current_chat_history) == 2:
            st.rerun()
