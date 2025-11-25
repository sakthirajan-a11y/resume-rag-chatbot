# app.py
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter   # ← FIXED
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils.pdf_parser import extract_text_from_pdf, extract_text_from_txt
import os
import tempfile

st.set_page_config(page_title="Resume RAG Chatbot", page_icon="Resume")
st.title("Resume RAG Chatbot")

# Sidebar
with st.sidebar:
    st.header("Settings")
    llm_choice = st.selectbox("LLM", ["Ollama (Llama 3.1)", "OpenAI (gpt-4o-mini)"])
    model_name = st.text_input("Ollama Model", value="llama3.1:8b") if "Ollama" in llm_choice else "gpt-4o-mini"

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload
uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Processing resume..."):
        # Save to temp file
        suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Extract text
        text = extract_text_from_pdf(tmp_path) if suffix == ".pdf" else extract_text_from_txt(tmp_path)

        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(text)

        # Embeddings & Vectorstore
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        st.session_state.vectorstore = FAISS.from_texts(splits, embeddings)

        os.unlink(tmp_path)  # delete temp file
        st.success("Resume loaded! Ask anything.")

# Chat
if st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    # LLM
    if "Ollama" in llm_choice:
        llm = ChatOllama(model=model_name, temperature=0)
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Prompt
    template = """You are an expert HR analyst. Answer ONLY based on the resume below.
    If information is not present, say "Not mentioned in the resume."

    Context: {context}

    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chat history
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input("Ask about the resume..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
                st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a resume to start chatting!")

st.caption("Try: Latest job title? • Technical skills? • Years of experience? • Contact email?")
