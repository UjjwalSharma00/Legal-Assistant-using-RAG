# Legal Assistant using RAG
A Retrieval-Augmented Generation (RAG) system that leverages LLM and RAG and Streamlit to let you chat with your own PDFs in your browser.

# About
This project is a Retrieval-Augmented Generation (RAG) system designed to help you interact with your own legal or patent PDFs using a conversational interface. Here’s how it works:

1. **User Interface:**  
   The app provides a simple web interface (built with Streamlit).

2. **PDF Upload (Optional):**  
   You can upload legal or patent PDFs through the interface. Uploaded files are stored and indexed for fast searching.

3. **Text Extraction:**  
   The system uses `pdfplumber` to extract text from each PDF.

4. **Text Chunking:**  
   Extracted text is split into manageable chunks using a text splitter. This helps the system efficiently search and retrieve relevant information.

5. **Embedding:**  
   Each text chunk is converted into vector embeddings using the Deepseek model (R1) via Ollama.

6. **Indexing:**
   Indexing is done by generating embeddings with OllamaEmbeddings

7. **Vector Store (FAISS)**
    The generated embeddings are indexed and stored in a vector store, enabling efficient similarity search and retrieval.
    The FAISS library is used to cluster and index the embeddings, allowing for fast and scalable semantic search across all document chunks.

8. **Retrieval:**  
   When you ask a question, the system searches the indexed chunks for the most relevant content using semantic similarity (Cosine similarity search).

9. **Answer Generation:**  
   The retrieved context is sent to a Large Language Model (LLM), which generates a detailed and comprehensive answer based on your question and the relevant document content. Using the answer template.

10. **History & Download:**  
   All your questions and answers are saved in a searchable history, which you can browse, paginate, and download as a CSV file.


## Pre-requisites
1. **Install Ollama** on your local machine from the [official website](https://ollama.com/).
2. Pull the Deepseek model:

    ```bash
    ollama pull deepseek-r1:1.5b
    ```
3. **Install dependencies** using pip:

    ```bash
    pip install -r requirements.txt
    ```

## SETUP
STEP - 1 - Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the Deepseek model:

STEP - 2 - type "ollama pull deepseek-r1:1.5b" in terminal

STEP - 3 - type "pip install -r requirements.txt" in terminal

STEP - 4 - run file or type "streamlit run pdf_rag.py" in terminal


## Run
You can run the Streamlit app in two ways:

### 1. Using the batch script (recommended for Windows):
```bat
run_app.bat
```

### 2. Directly with Streamlit:
```bash
streamlit run rag_V-7.py
```


## Usage

- **Upload PDFs:** Click the upload button to add your legal or patent PDFs. Uploaded files are indexed for fast searching.
- **Filter by Document:** Use the dropdown to filter your search to a specific uploaded or preloaded PDF.
- **Ask Questions:** Use the chat input at the bottom to ask questions about your documents. The assistant will use the content of your PDFs to answer.
- **View Search History:** Click "View Search History" to see previous questions and answers, with pagination for easy browsing.
- **Download History:** Download your entire search history as a CSV file for your records.
- **Clear Uploaded Files:** Remove all uploaded PDFs by clicking "Clear Uploaded Files" and confirming.
- **Preloaded PDFs:** Place any PDFs you want indexed by default in the `preloaded_pdfs/` folder before first run.
- **Security:** (Optional) Enable password protection in the code for restricted access.
