# Content Engine - Alemeno

A Streamlit application for processing PDF documents, storing them in a ChromaDB vector store, and enabling AI-driven question-answering.

## Features

- **PDF Processing**: Upload and process PDF documents into text chunks.
- **Vector Storage**: Store and query document embeddings with ChromaDB for efficient semantic search.
- **AI-Powered Q&A**: Generate detailed, context-based responses using AI models.
- **Document Re-Ranking**: Enhance response accuracy with CrossEncoder-based document relevance scoring.

### 1. Install the required dependencies:

pip install -r requirements.txt

### 2. Additional Downloads:

- Ollama Download
- Ollama Llama3.2:3b
- Ollama nomic-embed-text
- Cross Encoder Model


### 3. Run the Streamlit application:

streamlit run app_new.py

### 4. Process PDF Documents:

- Upload PDF files via the sidebar.
- Click "Process" to extract and store text chunks in the ChromaDB vector store.

### 5. Ask Questions:

- Enter a query in the text area.
- Click "Ask" to generate AI-driven responses based on the document content.
