import os
import logging
import streamlit as st
st.set_page_config(page_title="Almost Attorney", page_icon="⚖️")
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from concurrent.futures import ThreadPoolExecutor

# --- User Authentication ---
def login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("Login")
        password = st.text_input("Enter password", type="password")
        if st.button("Login"):
            if password == "your_secret_password":  # Change this to your desired password
                st.session_state.authenticated = True
                st.success("Logged in!")
            else:
                st.error("Incorrect password")
        st.stop()

# login()

# --- UI and Model Setup ---
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
template = """
You are an assistant for legal question-answering tasks. Use the retrieved context to answer the question. If you don't know, say that you don't know. 
Provide a detailed and comprehensive answer, using as much context as needed. Be clear and thorough in your explanation.
Question: {question} 
Context: {context} 
Answer:
"""
PRELOADED_PDFS_DIR = 'preloaded_pdfs/'
UPLOADED_PDFS_DIR = 'uploaded_pdfs/'
vector_store = None
FAISS_INDEX_PATH = "faiss_index/"


def get_vector_store():
    global vector_store
    if vector_store is None:
        embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        faiss_index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        if os.path.exists(faiss_index_file):
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            # Try to index PDFs only if there are any
            pdf_files = [f for f in os.listdir(PRELOADED_PDFS_DIR) if f.endswith(".pdf")]
            if not pdf_files:
                st.warning("No PDFs found in preloaded_pdfs/. Please add PDFs to index.")
                vector_store = None
            else:
                new_docs = []
                for pdf_file in pdf_files:
                    file_path = os.path.join(PRELOADED_PDFS_DIR, pdf_file)
                    try:
                        documents = load_pdf(file_path)
                        chunked_documents = split_text(documents)
                        new_docs.extend(chunked_documents)
                    except Exception as e:
                        print(f"Failed to process {pdf_file}: {e}")
                if new_docs:
                    vector_store = FAISS.from_documents(new_docs, embeddings)
                    vector_store.save_local(FAISS_INDEX_PATH)
                else:
                    st.warning("No valid content found in PDFs. Vector store not created.")
                    vector_store = None
    return vector_store

model = OllamaLLM(model="deepseek-r1:1.5b")

def load_custom_css():
    if os.path.exists("styles.css"):
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# load_custom_css()

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def load_and_index_pdfs(pdf_dir):
    if not os.path.exists(pdf_dir):
        raise ValueError(f"Directory {pdf_dir} does not exist.")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    new_docs = []
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_dir, pdf_file)
        try:
            documents = load_pdf(file_path)
            chunked_documents = split_text(documents)
            new_docs.extend(chunked_documents)
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")
    if new_docs:
        get_vector_store().add_documents(new_docs)
        get_vector_store().save_local(FAISS_INDEX_PATH)
        st.cache_data.clear()  # Clear cache after updating FAISS index

def upload_pdf(file, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not file.name.endswith(".pdf"):
        raise ValueError("Only PDF files are allowed.")
    file_path = os.path.join(directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def split_and_index_single_pdf(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunked_documents = text_splitter.split_documents(documents)
    get_vector_store().add_documents(chunked_documents)
    get_vector_store().save_local(FAISS_INDEX_PATH)
    st.cache_data.clear()  # Clear cache after updating FAISS index
    return chunked_documents

if not os.path.exists(PRELOADED_PDFS_DIR):
    os.makedirs(PRELOADED_PDFS_DIR)
if not os.path.exists(UPLOADED_PDFS_DIR):
    os.makedirs(UPLOADED_PDFS_DIR)

# Only index PDFs if the FAISS index file does not exist
faiss_index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
if not os.path.exists(faiss_index_file):
    try:
        load_and_index_pdfs(PRELOADED_PDFS_DIR)
        st.success("Preloaded PDFs have been successfully indexed into the vector store.")
    except Exception as e:
        st.error(f"Failed to index preloaded PDFs: {e}")
else:
    st.info("FAISS index found. Skipping PDF re-indexing for faster startup.")

# --- Advanced Search & Filtering  ---
def get_all_pdf_names():
    files = []
    for d in [PRELOADED_PDFS_DIR, UPLOADED_PDFS_DIR]:
        if os.path.exists(d):
            files.extend([f for f in os.listdir(d) if f.endswith(".pdf")])
    return files

def get_faiss_index_mtime():
    faiss_index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    return os.path.getmtime(faiss_index_file) if os.path.exists(faiss_index_file) else 0

@st.cache_data
def retrieve_docs(query, top_k=10, index_mtime=None):  # Increase top_k for more context
    results = get_vector_store().similarity_search_with_score(query)
    return [doc for doc, score in results[:top_k]]

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# --- Streamlit App UI ---
st.title("Legal Assistant - Patent RAG Model")

uploaded_file = st.file_uploader(
    "Upload your patent or legal PDF here:",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    with st.spinner("Processing your file..."):
        try:
            file_path = upload_pdf(uploaded_file, UPLOADED_PDFS_DIR)
            documents = load_pdf(file_path)
            chunked_documents = split_and_index_single_pdf(documents)
            st.success(f"Uploaded and indexed: {uploaded_file.name}")
            st.info(f"Number of documents indexed: {len(chunked_documents)}")
        except ValueError as ve:
            st.error(f"Error: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

search_history = st.session_state.get("search_history", [])

# --- Advanced Search & Filtering UI  ---
pdf_names = get_all_pdf_names()
selected_pdf = st.selectbox("Filter search by document (optional):", ["All"] + pdf_names)
selected_pdf = None if selected_pdf == "All" else selected_pdf

question = st.chat_input("Ask a legal question...")

if question:
    st.chat_message("user").write(question)
    related_documents = retrieve_docs(question, index_mtime=get_faiss_index_mtime())  # Pass index_mtime for cache invalidation
    answer = answer_question(question, related_documents)
    st.chat_message("assistant").write(answer)
    search_history.append({"question": question, "answer": answer})
    st.session_state.search_history = search_history

if st.button("View Search History"):
    page_size = 5
    total_pages = (len(search_history) + page_size - 1) // page_size
    page = st.number_input("Page", min_value=1, max_value=total_pages, step=1) - 1
    start = page * page_size
    end = start + page_size
    for entry in search_history[start:end]:
        st.write(f"**Question:** {entry['question']}")
        st.write(f"**Answer:** {entry['answer']}")

if search_history:
    import io
    import csv
    download_data = io.StringIO()
    writer = csv.DictWriter(download_data, fieldnames=["question", "answer"])
    writer.writeheader()
    for entry in search_history:
        writer.writerow({"question": entry["question"], "answer": entry["answer"]})
    st.download_button(
        label="Download history",
        data=download_data.getvalue(),
        file_name="search_history.csv",
        mime="text/csv"
    )

if st.button("Clear Uploaded Files"):
    if st.button("Confirm Clear"):
        for file in os.listdir(UPLOADED_PDFS_DIR):
            file_path = os.path.join(UPLOADED_PDFS_DIR, file)
            os.remove(file_path)
        st.success("All uploaded files have been cleared.")

st.markdown("---")
st.markdown("© 2025 Almost Attorney | Contact: support@example.com")