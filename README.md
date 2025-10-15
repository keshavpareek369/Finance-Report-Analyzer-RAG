# Multi-PDF RAG Chatbot with Streamlit and LangChain

This project is a web-based chatbot application built with Streamlit and LangChain. It allows users to upload multiple PDF documents and ask questions about their content. The application uses a Retrieval-Augmented Generation (RAG) architecture to provide context-aware answers by leveraging Google's Gemini LLM.

## 🚀 Features

* **Multi-PDF Upload**: Upload one or more PDF files simultaneously.
* **Session-Based Processing**: PDFs are processed only once per user session, ensuring efficiency.
* **Interactive Chat**: A user-friendly chat interface to ask questions about the uploaded documents.
* **Chat History**: Displays the conversation history for easy reference.
* **Efficient Context Management**: Implements a summary-based memory to keep track of the conversation, reducing API token usage.

***

## ⚙️ How It Works: The Project Flow

The application follows a sequential process from file upload to answer generation. This flow is orchestrated primarily by the `app.py` script, which calls helper functions from `loaders.py` and `embeddings_llm.py`.

### 1. File Upload & Initialization (app.py)

The user journey begins in the Streamlit web interface.

* **UI**: A file uploader component (`st.file_uploader`) allows the user to select and upload their PDF files.
* **Session State**: The application initializes a `session_state` to store the chat history, the vector store, the retriever, and the agent executor. This ensures that data persists across user interactions within the same session and prevents reprocessing of PDFs.

### 2. PDF Loading and Chunking (loaders.py)

Once files are uploaded and if no vector store exists in the current session, the processing pipeline begins.

* **Temporary Storage**: Each uploaded PDF is saved to a temporary file on the server using Python's `tempfile` library.
* **Document Loading**: The `load_pdf()` function in `loaders.py` uses `PyPDFLoader` from LangChain to load the text content from each PDF.
* **Text Splitting**: The loaded documents are passed to the `split_documents()` function. It uses `RecursiveCharacterTextSplitter` to break down the large texts into smaller, manageable chunks. This is a critical step in RAG, as it allows the model to work with smaller, more relevant pieces of context.

### 3. Embedding and Vector Store Creation (embeddings_llm.py)

With the text chunks ready, the next step is to convert them into a format that can be easily searched.

* **Embedding Model**: The `create_vectorstore()` function uses the `HuggingFaceEmbeddings` library to load a pre-trained sentence-transformer model (`all-mpnet-base-v2`). This model converts each text chunk into a high-dimensional numerical vector (an embedding).
* **Vector Store**: The embeddings and their corresponding text chunks are stored in a `Chroma` vector store. This specialized database is highly efficient at searching for vectors based on semantic similarity. The created vector store is then saved to the session state.

### 4. Agent and Tool Initialization (embeddings_llm.py & app.py)

After the vector store is ready, the application sets up an intelligent agent that can use this data.

* **Retriever**: A retriever is created from the Chroma vector store. Its sole job is to fetch the most relevant text chunks when given a query.
* **Retriever Tool**: This retriever is wrapped into a `create_retriever_tool`. This tool gives a name (`pdf_search`) and a description to the retriever, allowing the LangChain agent to understand what it does and when to use it.
* **LLM Initialization**: The `initialize_agent()` function configures the Large Language Model, in this case, Google's `gemini-2.0-flash-001`, using `ChatGoogleGenerativeAI`.
* **Agent Creation**: A **ReAct (Reasoning and Acting)** agent is created. This type of agent can reason about what tools it needs to use to answer a question. It's given the LLM, the retriever tool, and a prompt from LangChain Hub that guides its reasoning process. The final `AgentExecutor` is saved to the session state.

### 5. Chat Interaction and Query Execution (app.py)

With the agent ready, the user can now ask questions.

* **User Input**: The user types a question into the `st.chat_input` box.
* **Contextual Prompting**: The user's new question is combined with a running summary of the conversation history (`st.session_state['chat_summary']`). This gives the agent context about the ongoing conversation without needing to re-send the entire chat, which saves on API costs.
* **Agent Invocation**: The combined prompt is sent to the agent executor via the `run_query()` function.

### 6. The ReAct Agent's Thought Process

This is where the magic of LangChain's agent happens.

1.  **Thought**: The agent receives the prompt (e.g., "Chat summary: ... New question: What was the total revenue in 2023?"). It analyzes the new question and determines that it needs information from the uploaded documents.
2.  **Action**: The agent decides to use the `pdf_search` tool.
3.  **Action Input**: It passes the user's question to the tool.
4.  **Observation**: The tool (our retriever) searches the Chroma vector store and returns the most relevant text chunks related to "total revenue in 2023".
5.  **Thought**: The agent now has the original question and the retrieved context. It determines it has enough information to formulate a final answer.
6.  **Final Answer**: The agent passes the context and the question to the Gemini LLM, which generates a comprehensive, human-readable answer based *only* on the provided information.

### 7. Displaying the Response and Updating State (app.py)

* **Display Answer**: The final answer from the agent is captured and displayed on the Streamlit UI.
* **Update History**: The user's question and the agent's answer are appended to the `st.session_state["messages"]` list to maintain the chat history display.
* **Update Summary**: The conversation summary is updated with the latest exchange using the `update_summary()` function. This keeps the summary fresh for the next user query.

***

## 🔧 How to Set Up and Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Create a `.env` file:**
    Create a file named `.env` in the root directory and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

5.  Open your browser and navigate to the local URL provided by Streamlit.
