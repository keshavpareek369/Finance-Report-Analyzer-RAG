# embeddings_llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def create_vectorstore(chunks, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
    """Create Chroma vectorstore from document chunks."""


    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    vectorstore = Chroma.from_documents(documents=chunks, embedding=hf)
    return vectorstore



def initialize_agent(tools, llm_model: str = "gemini-2.0-flash-001"):
    """Initialize a LangChain ReAct agent with given tools and LLM."""
    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")  # ✅ explicitly use your API key
    )
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_query(agent_executor, query: str):
    """Run a query through the agent executor."""
    return agent_executor.invoke({"input": query})
