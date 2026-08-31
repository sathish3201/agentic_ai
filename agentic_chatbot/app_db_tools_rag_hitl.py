from agentic_chatbot_db_tools_rag_hitl import ingest_rag_documents,chatbot, get_all_threads, delete_thread,get_retriever, set_last_uploaded_jd
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import streamlit as st
import uuid
import tempfile
import os
import re
from langgraph.types import Command
# -------------------------------------

# Matches the ===FINAL_TAILORED_RESUME_START/END=== markers that
# resume_review_tool asks the LLM to wrap its final tailored resume text
# in, so this UI can offer it as a downloadable .txt/.md file.
_TAILORED_RESUME_PATTERN = re.compile(
    r"===FINAL_TAILORED_RESUME_START===\s*(.*?)\s*===FINAL_TAILORED_RESUME_END===",
    re.DOTALL,
)


def render_resume_download_if_present(content, key_suffix):
    """If the assistant's message contains a delimited tailored-resume
    block, strip the delimiters out of the visible text and show a
    download button for it (.txt for now; .docx/.pdf come from the
    offer_file_export HITL step)."""
    match = _TAILORED_RESUME_PATTERN.search(content)
    if not match:
        return content

    resume_text = match.group(1).strip()
    display_text = _TAILORED_RESUME_PATTERN.sub(
        "*(Tailored resume ready -- see download button below.)*", content
    )

    st.download_button(
        "⬇️ Download tailored resume (.txt)",
        data=resume_text,
        file_name="tailored_resume.txt",
        mime="text/plain",
        key=f"download_resume_{key_suffix}",
    )

    return display_text
# generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())

#prevents the same thread id from being used multiple times
def add_thread(thread_id):
    """Add a new thread ID if it doesn't exist"""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def save_uploaded_file(uploaded_file):

    temp_dir = tempfile.gettempdir()

    file_path = os.path.join(
        temp_dir,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    return file_path

def _make_title(text):
    """Trim a message down to a short, sidebar-friendly title."""
    title = text.strip().splitlines()[0]
    if len(title) > 40:
        title = title[:40].rstrip() + "..."
    return title


def get_thread_title(thread_id):
    """Return a display title for a thread, deriving it from the thread's
    first stored user message (and caching it) if not already known."""
    titles = st.session_state['thread_titles']
    if thread_id not in titles:
        for message in load_conversations(thread_id):
            if isinstance(message, HumanMessage) and message.content:
                titles[thread_id] = _make_title(message.content)
                break
    # no messages sent yet on this thread -- show a friendly placeholder
    # instead of the raw UUID. Not cached in `titles`, so the very next
    # call (e.g. right after the first message is sent) picks up the
    # real derived title automatically instead of staying stuck here.
    return titles.get(thread_id, "New Thread")


def set_thread_title_from_message(thread_id, message):
    """Derive a short chat title from the first user message, once."""
    if thread_id not in st.session_state['thread_titles']:
        st.session_state['thread_titles'][thread_id] = _make_title(message)

#reset 
def reset_chat():
    """Reset the chat"""
    st.session_state['thread_id'] = generate_thread_id()

    st.session_state['message_history'] = []
    add_thread(st.session_state['thread_id'])


# loading conversations
def load_conversations(thread_id):
    """Load conversation from thread"""
    state=chatbot.get_state(config={'configurable':{'thread_id':thread_id}})
    # return emptyempty not emp
    return state.values.get("messages", [])


st.set_page_config(page_title="Agentic AI ChatBot", page_icon="🤖", layout="wide")

# ============ LAYOUT + STYLING ===============================================
st.markdown(
    """
    <style>
    /* ---- overall canvas: give the chat room to breathe, cap it at a
       comfortable reading width, and keep it centered instead of pinned
       to a cramped 700px column or stretched edge-to-edge.
       Uses clamp() so it scales down smoothly on narrower windows
       instead of snapping between fixed breakpoints. ---- */
    .block-container {
        max-width: min(880px, 92vw);
        margin: 0 auto;
        padding-top: clamp(1rem, 3vw, 2rem);
        padding-bottom: 6rem;
        padding-left: clamp(0.75rem, 3vw, 2rem);
        padding-right: clamp(0.75rem, 3vw, 2rem);
    }

    /* small screens: sidebar overlays instead of splitting the viewport,
       so the chat column keeps a usable width on phones/tablets */
    @media (max-width: 640px) {
        .block-container {
            max-width: 100vw;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
    }

    /* ---- header ---- */
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.25rem;
    }
    .app-header .icon {
        font-size: 2rem;
        line-height: 1;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    .app-subtitle {
        color: #6B7280;
        font-size: 0.95rem;
        margin: 0 0 1.75rem 0;
    }

    /* ---- sidebar: wider, padded, visually separated ---- */
    section[data-testid="stSidebar"] {
        min-width: 300px;
        background: #FAFAFC;
        border-right: 1px solid #ECECF1;
    }
    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    section[data-testid="stSidebar"] button {
        border-radius: 10px;
        width: 100%;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    section[data-testid="stSidebar"] .stButton {
        margin-bottom: 0.4rem;
    }

    /* ---- chat messages: more vertical breathing room between turns ---- */
    div[data-testid="stChatMessage"] {
        margin-bottom: 1.1rem;
        gap: 0.75rem;
        max-width: 100%;
    }

    /* Streamlit nests the actual text several levels deep inside its own
       auto-layout wrappers (stChatMessageContent > stVerticalBlockBorderWrapper
       > ... > stMarkdown). Those inner wrappers shrink-wrap to their own
       content, which is what made the bubble collapse to a narrow column
       instead of using the free width of the row. Forcing every wrapper in
       that chain to 100% width lets the outer bubble (stMarkdownContainer)
       be the only thing that controls sizing. */
    div[data-testid="stChatMessageContent"] div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stChatMessageContent"] div[data-testid="stVerticalBlock"],
    div[data-testid="stChatMessageContent"] div[data-testid="element-container"],
    div[data-testid="stChatMessageContent"] div[data-testid="stMarkdown"] {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* user messages: right-aligned, avatar on the right, blue bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"])
        div[data-testid="stChatMessageContent"] {
        display: flex;
        justify-content: flex-end;
        flex: 1 1 auto;
        min-width: 0;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"])
        div[data-testid="stMarkdownContainer"] {
        background: #DBEAFE;
        border-radius: 18px 18px 4px 18px;
        padding: 0.7rem 1rem;
        max-width: clamp(240px, 85%, 640px);
        width: fit-content;
        text-align: left;
    }

    /* assistant messages: soft grey bubble, rounded on the left */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"])
        div[data-testid="stChatMessageContent"] {
        display: flex;
        justify-content: flex-start;
        flex: 1 1 auto;
        min-width: 0;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"])
        div[data-testid="stMarkdownContainer"] {
        background: #F3F4F6;
        border-radius: 18px 18px 18px 4px;
        padding: 0.7rem 1rem;
        max-width: clamp(240px, 85%, 640px);
        width: fit-content;
    }

    /* ---- "thinking / tool status" card ---- */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #E6E6EA;
        background: #FCFCFD;
    }

    /* ---- chat input: wider, more padding, sits nicely above the fold ---- */
    div[data-testid="stChatInput"] textarea {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <span class="icon">🤖</span>
        <h1>Agentic AI ChatBot</h1>
    </div>
    <p class="app-subtitle">Built with LangGraph · web search, calculator, stock &amp; weather tools</p>
    """,
    unsafe_allow_html=True,
)

#loading message history from previous conversations
if 'message_history' not in st.session_state:
    st.session_state['message_history']= []

#list of thread ids
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']= get_all_threads()

#thread_id -> display title
if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    # add_thread(st.session_state['thread_id'])


# ADD CURRENT THREAD TO CONVERSATION LIST
add_thread(st.session_state['thread_id'])

# ======= sidebar  threading feature title ==============================

st.sidebar.title("💬 My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):

    #reset the current chat and create a new thread
    reset_chat()

    #rerun the streamlit app to update the interface
    st.rerun()

st.sidebar.markdown("<div style='margin-bottom:0.75rem'></div>", unsafe_allow_html=True)


# =============DISPLAYING ALL CONVERSATIONS IN REVERSE ORDER ======
for thread_id in st.session_state['chat_threads'][::-1]:

    is_active = thread_id == st.session_state['thread_id']

    #ONE ROW PER CONVERSATION: select button + delete button
    col_select, col_delete = st.sidebar.columns([6, 1], gap="small")

    with col_select:
        label = ("🟢 " if is_active else "") + get_thread_title(thread_id)
        if st.button(label, key=thread_id, use_container_width=True):

            # SET SELECTED THREAD AS THE CURRENT THREAD
            st.session_state['thread_id'] = thread_id

            # load messages SAVED UNDER THE SELECTED THREAD
            messages = load_conversations(thread_id)
            temp_messages = []
            for message in messages:
                # check whether message sent by user
                if isinstance(message, HumanMessage):
                    role = 'user'

                elif isinstance(message, AIMessage):
                    role = 'assistant'
                # ignore other message types, such as ToolMessage
                else:
                    continue
                temp_messages.append({'role': role, 'content': message.content})
            # save the conversation to session state
            # st.session_state[f'thread_{thread_id}'] = temp_messages
            st.session_state['message_history'] = temp_messages
            #RERUN THE APPLICATION TO DISPALY THE LOADED MESSAGES
            st.rerun()

    with col_delete:
        if st.button("🗑️", key=f"delete_{thread_id}"):
            delete_thread(thread_id)
            st.session_state['chat_threads'].remove(thread_id)
            st.session_state['thread_titles'].pop(thread_id, None)

            # if the deleted thread was active, switch to a fresh new chat
            if st.session_state['thread_id'] == thread_id:
                reset_chat()

            st.rerun()

# ===========================MAIN CHAT INTERFACE ==================

#display all messages from currently selected conversation
for idx, message in enumerate(st.session_state['message_history']):
    #create either user chatb bubble or assistant chat bubble
    with st.chat_message(message['role']):

        #dispaly message conetent
        content = message['content']
        if message['role'] == 'assistant':
            content = render_resume_download_if_present(content, key_suffix=f"history_{idx}")
        st.markdown(content)

# -----------------------------hitl user interface-----------

# Interrupt state must live in session_state, NOT be gated behind
# `if user_input:` -- st.chat_input() only returns truthy on the exact
# rerun right after the user submits a message. Clicking the Approve/
# Reject button below triggers its OWN rerun, on which user_input is
# None again, so any interrupt/resume logic nested under `if user_input:`
# would simply never run and the app would look frozen after the click.
if 'pending_hitl' not in st.session_state:
    st.session_state['pending_hitl'] = None  # {"question": ..., "config": {...}}

if st.session_state['pending_hitl']:

    pending = st.session_state['pending_hitl']
    step = pending.get("step")

    st.warning("⚠️ Human input required")
    st.markdown(f"### {pending['question']}")

    decision = None

    if step == "offer_file_export":
        # this checkpoint expects a format choice, not a yes/no --
        # render buttons matching what its registered handler
        # (_hitl_offer_file_export) actually understands
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📄 DOCX", key=f"docx_{st.session_state['thread_id']}", use_container_width=True):
                decision = "docx"
        with col2:
            if st.button("📑 PDF", key=f"pdf_{st.session_state['thread_id']}", use_container_width=True):
                decision = "pdf"
        with col3:
            if st.button("🚫 No, skip", key=f"skip_{st.session_state['thread_id']}", use_container_width=True):
                decision = "no"
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve", key=f"approve_{st.session_state['thread_id']}", use_container_width=True):
                decision = "approved"
        with col2:
            if st.button("❌ Reject", key=f"reject_{st.session_state['thread_id']}", use_container_width=True):
                decision = "rejected"

    if decision is not None:

        resume_config = pending["config"]

        # clear it now so a rerun mid-resume doesn't re-show stale buttons
        st.session_state['pending_hitl'] = None

        with st.chat_message("assistant"):

            response_placeholder = st.empty()
            resume_response = ""

            with st.status("▶️ Resuming...", expanded=False) as resume_status:

                for mode, chunk in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode=["messages", "updates", "custom"],
                ):

                    if mode == "custom":
                        progress = chunk.get("progress_update")
                        if progress:
                            resume_status.update(label=progress["label"])
                        continue

                    if mode == "messages":
                        message_chunk, _metadata = chunk
                        if isinstance(message_chunk, AIMessage) and message_chunk.content:
                            if isinstance(message_chunk.content, str):
                                resume_response += message_chunk.content
                                response_placeholder.markdown(resume_response + "▌")
                        continue

                    if "__interrupt__" in chunk:
                        interrupts = chunk["__interrupt__"]
                        if interrupts:
                            # another checkpoint is needed (e.g. the
                            # ats_statement -> offer_file_export chain
                            # dynamically triggered by the HITL step
                            # registry) -- stash it and stop; the button UI
                            # above renders the right controls based on
                            # "step" on the next rerun
                            interrupt_value = interrupts[0].value
                            st.session_state['pending_hitl'] = {
                                "question": (
                                    interrupt_value.get("question", "Do you want to continue?")
                                    if isinstance(interrupt_value, dict)
                                    else str(interrupt_value)
                                ),
                                "step": (
                                    interrupt_value.get("step")
                                    if isinstance(interrupt_value, dict)
                                    else None
                                ),
                                "config": resume_config,
                            }
                            break

                    # AI text content now comes from the "messages" branch
                    # above; this loop is only used for ToolMessage status.
                    for node_name, node_update in chunk.items():

                        if not isinstance(node_update, dict):
                            continue

                        messages = node_update.get("messages", [])

                        for message in messages:

                            if isinstance(message, ToolMessage):
                                resume_status.update(label=f"✅ `{message.name}` completed")

                resume_status.update(label="✅ Done", state="complete", expanded=False)

            response_placeholder.markdown(resume_response)

            if resume_response:
                st.session_state["message_history"].append(
                    {"role": "assistant", "content": resume_response}
                )

        st.rerun()

user_input = st.chat_input(
    placeholder="Type Here...",
    accept_file=True,
    file_type=["pdf", "docx", "txt"],
    max_upload_size=200,
)

if user_input:

    user_message = user_input.text

    # -----------------------------------------------------
    # Handle uploaded files
    # -----------------------------------------------------
    if user_input.files:

        uploaded_file = user_input.files[0]

        file_path = save_uploaded_file(uploaded_file)

        # if the message mentions "job description"/"JD"/"job posting",
        # treat this upload as the JD rather than the resume -- otherwise
        # it's ingested as the resume, same as before
        jd_keywords = ("job description", " jd ", " jd.", " jd,", "jd:", "job posting")
        is_jd_upload = any(kw in f" {user_message.lower()} " for kw in jd_keywords)

        if is_jd_upload:
            set_last_uploaded_jd(file_path)
        else:
            ingest_rag_documents(file_path)

        # a user attaching a file with NO typed text is common (attach +
        # send). Leaving user_message empty here would mean: (1) nothing
        # shows in the user's own chat bubble, and (2) the LLM gets an
        # empty HumanMessage with no instruction, so it just asks "please
        # share the file" again -- even though it was already ingested.
        # Synthesize a concrete default message naming the actual file.
        if not user_message.strip():
            user_message = (
                f"I've uploaded the job description file \"{uploaded_file.name}\"."
                if is_jd_upload else
                f"I've uploaded my resume \"{uploaded_file.name}\". Please review it."
            )

    # -----------------------------------------------------
    # Save user message
    # -----------------------------------------------------
    set_thread_title_from_message(
        st.session_state["thread_id"],
        user_message
    )

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------
    with st.chat_message("user"):
        st.markdown(user_message)

    # -----------------------------------------------------
    # LangGraph config
    # -----------------------------------------------------
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    # -----------------------------------------------------
    # Assistant response
    # -----------------------------------------------------
    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""

        # Flag to determine whether LangGraph interrupted
        hitl_interrupt = None

        with st.status(
            "🤔 Thinking...",
            expanded=False
        ) as status:

            # -------------------------------------------------
            # Stream graph -- "updates" for normal graph state changes,
            # "custom" for the live progress/ETA events tools emit via
            # report_progress()/get_stream_writer() (see backend's
            # dynamic progress-stage registry). With multiple stream
            # modes, each item is a (mode_name, payload) tuple.
            # -------------------------------------------------
            # "messages" gives real token-by-token streaming (AIMessageChunk
            # per token) -- without it, only "updates" is seen, which only
            # delivers each node's COMPLETE output after it finishes, so the
            # whole response appears at once instead of streaming in live.
            for mode, chunk in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_message)
                    ]
                },
                config=CONFIG,
                stream_mode=["messages", "updates", "custom"],
            ):

                if mode == "custom":
                    progress = chunk.get("progress_update")
                    if progress:
                        status.update(label=progress["label"])
                    continue

                if mode == "messages":
                    # chunk is (message_chunk, metadata) for "messages" mode
                    message_chunk, _metadata = chunk
                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        if isinstance(message_chunk.content, str):
                            full_response += message_chunk.content
                            response_placeholder.markdown(full_response + "▌")
                    continue

                # ---------------------------------------------
                # Check LangGraph interrupt
                # ---------------------------------------------
                if "__interrupt__" in chunk:

                    interrupts = chunk["__interrupt__"]

                    if interrupts:

                        hitl_interrupt = interrupts[0]

                        status.update(
                            label="⏸️ Waiting for your approval...",
                            state="running",
                            expanded=True,
                        )

                        break

                # ---------------------------------------------
                # Process graph updates -- only used for tool-call status
                # and ToolMessage events here; AI text content now comes
                # from the "messages" branch above instead, to avoid
                # double-appending the same content from both modes.
                # ---------------------------------------------
                for node_name, node_update in chunk.items():

                    if not isinstance(node_update, dict):
                        continue

                    messages = node_update.get("messages", [])

                    for message in messages:

                        # -----------------------------
                        # AI message
                        # -----------------------------
                        if isinstance(message, AIMessage):

                            # Tool calls
                            if message.tool_calls:

                                for tool_call in message.tool_calls:

                                    tool_name = tool_call["name"]

                                    status.update(
                                        label=f"🔎 Starting `{tool_name}`..."
                                    )

                        # -----------------------------
                        # Tool message
                        # -----------------------------
                        elif isinstance(message, ToolMessage):

                            status.update(
                                label=f"✅ `{message.name}` completed"
                            )

            # -------------------------------------------------
            # HITL INTERRUPT
            # -------------------------------------------------
            if hitl_interrupt:

                interrupt_value = hitl_interrupt.value

                status.update(
                    label="⏸️ Approval required",
                    state="running",
                    expanded=True,
                )

        # =====================================================
        # STASH HITL INTERRUPT (button UI + resume logic live in the
        # session_state-driven block above, so they survive the rerun
        # that happens the instant Approve/Reject is clicked)
        # =====================================================

        if hitl_interrupt:

            if isinstance(interrupt_value, dict):
                question = interrupt_value.get("question", "Do you want to continue?")
                step = interrupt_value.get("step")
            else:
                question = str(interrupt_value)
                step = None

            st.session_state['pending_hitl'] = {
                "question": question,
                "step": step,
                "config": CONFIG,
            }

            st.rerun()

        else:

            # =================================================
            # Normal completion
            # =================================================

            status.update(
                label="✅ Done",
                state="complete",
                expanded=False,
            )

            display_text = render_resume_download_if_present(
                full_response, key_suffix="live"
            )
            response_placeholder.markdown(
                display_text
            )

            st.session_state[
                "message_history"
            ].append(
                {
                    "role": "assistant",
                    "content": full_response,
                }
            )


