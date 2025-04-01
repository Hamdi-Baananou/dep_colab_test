# Graph RAG Document Assistant

This Streamlit application leverages a Retrieval-Augmented Generation (RAG) approach combined with a Neo4j knowledge graph to answer questions based on the content of uploaded PDF documents.

## Description

The application allows users to:
1.  Upload multiple PDF documents.
2.  Process these documents by extracting text, splitting it into chunks, and generating embeddings.
3.  Build a knowledge graph in Neo4j, identifying key entities (like Persons, Organizations, Locations) and concepts within the text and linking them.
4.  Store text chunks and their vector embeddings in Neo4j for semantic search.
5.  Ask natural language questions about the document content.
6.  Retrieve relevant context using both vector similarity search and graph traversal.
7.  Synthesize an answer using an OpenAI language model (GPT-4o by default) based on the retrieved context.

## Features

*   **PDF Upload & Processing:** Handles multiple PDF uploads, extracts text using PyMuPDF, and cleans it.
*   **Text Chunking:** Uses `RecursiveCharacterTextSplitter` for optimal chunking.
*   **Embedding Generation:** Employs `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face for creating text embeddings.
*   **Knowledge Graph Construction:**
    *   Connects to a Neo4j database.
    *   Creates `Document`, `Chunk`, `Entity`, and `Concept` nodes.
    *   Establishes relationships (`CONTAINS`, `MENTIONS`, `RELATED_TO`).
    *   Uses OpenAI API for entity/concept extraction from text chunks.
*   **Vector Indexing:** Stores and indexes chunk embeddings in Neo4j for efficient similarity search.
*   **Hybrid Retrieval:** Combines vector search and graph traversal (based on entities in the question) to find relevant context.
*   **LLM Integration:** Uses OpenAI's API (configurable model) to generate answers based on context.
*   **Streamlit Interface:** Provides a user-friendly web interface for interaction.
*   **Configuration:** Supports configuration via Streamlit secrets or sidebar inputs for API keys and database credentials.
*   **Debugging Tools:** Includes basic tools to inspect graph statistics and analyze query context retrieval.

## Requirements

*   Python 3.8+
*   Neo4j Database (v4.4+ recommended for vector index support, or AuraDB)
*   OpenAI API Key
*   Python packages listed in `requirements.txt`

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `sentence-transformers` might download models on first run, requiring internet access).*

3.  **Neo4j Setup:**
    *   Ensure you have a running Neo4j instance (Desktop, Server, or AuraDB).
    *   Note down the **URI**, **Username**, and **Password** for your database.
    *   Ensure the APOC plugin is installed if using Neo4j Server/Desktop, as it's often helpful (though vector indexes are the main requirement here).

4.  **OpenAI API Key:**
    *   Obtain an API key from the [OpenAI Platform](https://platform.openai.com/).

5.  **Configuration (Secrets - Recommended for Deployment):**
    *   If deploying via Streamlit Community Cloud, create a `secrets.toml` file in a `.streamlit` directory:
      ```toml
      # .streamlit/secrets.toml

      NEO4J_URI = "bolt://your_neo4j_host:7687" # or "neo4j+s://<your-aura-db-id>.databases.neo4j.io"
      NEO4J_USERNAME = "neo4j"
      NEO4J_PASSWORD = "your_neo4j_password"

      OPENAI_API_KEY = "sk-your_openai_api_key"
      ```
    *   For local development, you can still use secrets or enter credentials directly in the Streamlit sidebar.

## Usage

1.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

2.  **Configure:**
    *   Open the app in your browser (usually `http://localhost:8501`).
    *   Use the sidebar to enter your Neo4j credentials and OpenAI API Key if not using secrets.
    *   Click "Connect to Neo4j".

3.  **Upload & Process:**
    *   Upload one or more PDF files using the file uploader in the sidebar.
    *   Click the "Process Files & Build Graph" button in the main area. Wait for the processing to complete (this can take time depending on the number/size of PDFs and API latency).

4.  **Ask Questions:**
    *   Once processing is complete, go to the "Ask Questions" section.
    *   Enter your question about the content of the uploaded documents in the text area.
    *   Click "Get Answer". The application will retrieve context and generate an answer.

5.  **Debug (Optional):**
    *   Use the "Debugging & Analysis" section to view graph statistics or analyze the context retrieved for a specific query.

## Future Work / Improvements

*   More sophisticated entity/relationship extraction prompts.
*   Graph schema refinement based on specific use cases.
*   Enhanced graph traversal strategies for context retrieval.
*   User feedback mechanism for answer quality.
*   Support for other document types (e.g., .txt, .docx).
*   More advanced visualization of the knowledge graph.
*   Caching strategies for expensive operations (embeddings, LLM calls). 