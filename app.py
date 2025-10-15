# app.py
import streamlit as st
import tempfile
from loaders import load_pdf, split_documents
from embeddings_llm import create_vectorstore, initialize_agent, run_query
from langchain.tools.retriever import create_retriever_tool

st.set_page_config(page_title="📘 Multi-PDF RAG Chatbot", layout="wide")
st.title("🤖 Multi-PDF Finance Report Assistant")

# --- Sidebar Instructions / Features ---
with st.sidebar.expander("ℹ️ How to Use / Features", expanded=True):
    st.markdown("""
    **Features:**
    - Upload one or more PDFs at once.
    - PDFs are processed only once per session.
    - Chatbot can answer questions about uploaded PDFs.
    - Summary-based memory reduces API usage while keeping context.
    - Chat history is displayed above the input box.

    **How to Use:**
    1. Upload PDFs using the 'Browse files' button.
    2. Wait until processing is complete (progress shown below).
    3. Ask questions about your PDFs in the chat box.
    4. Responses appear below each question.
    """)

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []           # chat history
if "agent_executor" not in st.session_state:
    st.session_state["agent_executor"] = None
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "chat_summary" not in st.session_state:
    st.session_state["chat_summary"] = ""      # summary-based memory

# --- Upload PDFs ---
uploaded_files = st.file_uploader(
    "Upload one or more PDF files", 
    type=["pdf"], 
    accept_multiple_files=True
)

# ✅ Process PDFs only once per session
if uploaded_files and st.session_state["vectorstore"] is None:
    processing_msg = st.info("⏳ Upload detected! Processing PDFs, please wait...")
    all_chunks = []
    for uploaded_file in uploaded_files:
        # Save each PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and split
        docs = load_pdf(tmp_path)
        chunks = split_documents(docs)
        all_chunks.extend(chunks)

    # Build vectorstore once
    st.session_state["vectorstore"] = create_vectorstore(all_chunks)
    st.session_state["retriever"] = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 10})
    processing_msg.empty()  # Remove spinner message
    st.success("✅ All PDFs processed! You can start chatting below:")

# --- Initialize Agent ---
if st.session_state["retriever"] and st.session_state["agent_executor"] is None:
    tools = [
        create_retriever_tool(
            st.session_state["retriever"],
            name="pdf_search",
            description="Searches across the uploaded PDFs for relevant information"
        )
    ]
    st.session_state["agent_executor"] = initialize_agent(tools)

# --- Function to update chat summary ---
def update_summary(current_summary, user_question, assistant_answer, max_length=1000):
    """
    Create a short summary of the conversation to reduce token usage.
    Truncates summary to max_length characters.
    """
    new_summary = f"{current_summary}\nUser: {user_question}\nAssistant: {assistant_answer}"
    if len(new_summary) > max_length:
        new_summary = new_summary[-max_length:]
    return new_summary

# --- Chat interface (always visible) ---
st.subheader("💬 Chat with your PDFs")

# Display previous messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input new query (always visible, even if PDFs not uploaded yet)
query = st.chat_input("Ask a question about your PDFs...")

if query:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state["agent_executor"] is None:
        # Agent not ready yet (PDFs not uploaded)
        with st.chat_message("assistant"):
            st.markdown("⚠️ Please upload PDFs first to start getting answers.")
    else:
        # Prepare query using summary-based memory
        prompt_with_summary = f"Chat summary:\n{st.session_state['chat_summary']}\n\nNew question:\n{query}"

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                response = run_query(st.session_state["agent_executor"], prompt_with_summary)
                answer = response["output"]
                st.markdown(answer)

        # Save assistant message
        st.session_state["messages"].append({"role": "assistant", "content": answer})

        # Update summary
        st.session_state["chat_summary"] = update_summary(
            st.session_state["chat_summary"], query, answer
        )
