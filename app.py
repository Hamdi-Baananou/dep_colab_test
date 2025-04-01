# =============================
# Required imports
# =============================
import os
import re
import requests
import json
import logging
import time
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional
# import matplotlib.pyplot as plt # Matplotlib might be less useful in a basic webapp unless plotting graphs
import streamlit as st
import tempfile
import concurrent.futures # Add this import

# RAG imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Graph DB imports
from langchain_community.graphs import Neo4jGraph
# Note: langchain.vectorstores.neo4j_vector is deprecated. Use langchain_community.vectorstores.Neo4jVector
from langchain_community.vectorstores import Neo4jVector
# from langchain.chains import GraphQAChain # Not explicitly used in the final QA function, using custom logic
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
import requests

# =============================
# Original Code (Functions adapted for Streamlit)
# =============================

# Create a custom LLM class for DeepSeek
class DeepSeekLLM(LLM):
    api_key: str
    model: str = "deepseek-reasoner"  # Changed default to reasoner
    max_tokens: int = 1024
    temperature: float = 0.1
    messages: List[Dict[str, str]] = []  # Add message history
    api_base_url: str = "https://api.deepseek.com/v1"

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Add the new prompt to message history
        # Only add history if the class instance is intended for conversation
        # For the temporary client in create_knowledge_graph, this might not be desired
        # We'll keep it for now, but be aware it affects the temporary client too.
        self.messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # Use the current message history for the payload
            "messages": self.messages
        }

        if stop:
            payload["stop"] = stop

        request_url = f"{self.api_base_url}/chat/completions"

        try:
            # ADD TIMEOUT HERE (e.g., 60 seconds)
            response = requests.post(request_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        # CATCH TIMEOUT EXCEPTION
        except requests.exceptions.Timeout:
            logger.error(f"API Request timed out after 60 seconds for model {self.model}.")
            st.error("API request timed out. The service might be slow or unavailable.")
            # Re-raise or return an error indicator
            raise ValueError("API Timeout")
        except requests.exceptions.RequestException as e:
             st.error(f"API Request Error: {e}")
             logger.error(f"API Request Error: {e}", exc_info=True)
             # Clear messages potentially causing issues if request fails badly? Optional.
             # self.messages.pop() # Remove the last user message if it caused the error
             raise ValueError(f"API Error: {e}")

        response_data = response.json()

        if 'choices' not in response_data or len(response_data['choices']) == 0:
            logger.error(f"Invalid response format from API: {response_data}")
            raise ValueError("Invalid response format from API")

        first_choice = response_data['choices'][0]
        message_content = first_choice.get('message', {}).get('content', '')
        
        # Add assistant's response to message history
        # Only add history if the class instance is intended for conversation
        # For the temporary client, this might build up unnecessarily.
        self.messages.append({"role": "assistant", "content": message_content})

        return message_content

    def clear_history(self):
        """Clear the message history"""
        self.messages = []

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base_url": self.api_base_url
        }

# Setup Logging (Keep logging, useful for debugging in Streamlit Cloud logs)
class ColoredFormatter(logging.Formatter):
    COLORS = {'DEBUG': '\033[94m', 'INFO': '\033[92m', 'WARNING': '\033[93m', 'ERROR': '\033[91m', 'CRITICAL': '\033[91m\033[1m', 'ENDC': '\033[0m'}
    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}{self.COLORS['ENDC']}"

logger = logging.getLogger('graph_rag_streamlit')
logger.setLevel(logging.INFO) # Set to INFO for Streamlit Cloud to avoid overly verbose logs
# Clear existing handlers if any (important for Streamlit re-runs)
if logger.hasHandlers():
    logger.handlers.clear()

# Console handler (will show up in Streamlit Cloud logs)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Use standard formatter for cloud logs
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)
# Optional File handler (might not be easily accessible/persistent on Streamlit Cloud free tier)
# file_handler = logging.FileHandler('graph_rag_streamlit.log')
# file_handler.setLevel(logging.DEBUG)
# file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(file_format)
# logger.addHandler(file_handler)

# Initialize Embeddings (Using Hugging Face) - Cache this
@st.cache_resource # Cache the embedding model
def get_embeddings():
    logger.info("Initializing embedding model...")
    start_time = time.time()
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} # Ensure CPU usage for broader compatibility
        )
        logger.info(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")
        return embedding_function
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", exc_info=True)
        st.error(f"Error loading embedding model: {e}")
        return None

# PDF Processing Functions
# Adapt process_pdfs to take uploaded file objects
def process_pdfs_streamlit(uploaded_files):
    """Process uploaded PDF files with error handling, text cleaning, and chunking"""
    if not uploaded_files:
        st.warning("No PDF files uploaded.")
        return [], {}

    logger.info(f"Starting to process {len(uploaded_files)} PDF files")
    all_chunks = []
    source_metadata = {}
    temp_files = []

    # Use a context manager for temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            try:
                start_time = time.time()
                # Save uploaded file to a temporary path
                temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_files.append(temp_pdf_path) # Keep track if needed, but loader uses path

                logger.info(f"Processing {uploaded_file.name}")
                st.info(f"Processing {uploaded_file.name}...")

                # Load PDF
                loader = PyMuPDFLoader(temp_pdf_path)
                documents = loader.load()
                logger.debug(f"Loaded {len(documents)} pages from {uploaded_file.name}")

                # Store metadata
                source_name = uploaded_file.name # Use original filename
                source_metadata[source_name] = {
                    "filename": source_name,
                    "pages": len(documents),
                    "processed_at": datetime.now().isoformat()
                }

                # Clean text
                for doc in documents:
                    doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
                    doc.metadata['source'] = source_name

                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " ", ""],
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)

                # Add more detailed metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata['chunk_id'] = f"{source_name}_chunk_{i}"
                    chunk.metadata['source_document'] = source_name
                    # Ensure page number is captured if available
                    if 'page' not in chunk.metadata:
                         chunk.metadata['page'] = doc.metadata.get('page', 0) # Get page from parent doc if possible

                all_chunks.extend(chunks)
                logger.info(f"Processed {uploaded_file.name} into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")
                st.info(f"Finished processing {uploaded_file.name} ({len(chunks)} chunks).")

            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {str(e)}", exc_info=True)
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
            finally:
                 # Temporary file is automatically removed when temp_dir context exits
                 pass


    logger.info(f"Completed processing all PDFs. Generated {len(all_chunks)} total chunks")
    if not all_chunks:
        st.error("No text chunks could be extracted from the uploaded PDFs.")
    return all_chunks, source_metadata

# Neo4j Graph Functions
# @st.cache_resource # Caching connection can be tricky if creds change; connect on demand
def connect_to_neo4j(uri, username, password):
    """Establish connection to Neo4j and return graph object"""
    try:
        logger.info(f"Attempting to connect to Neo4j at {uri}")
        graph = Neo4jGraph(
            url=uri,
            username=username,
            password=password
        )
        # Test connection
        result = graph.query("MATCH (n) RETURN count(n) as count")
        logger.info(f"Successfully connected to Neo4j. Database has {result[0]['count']} nodes")
        st.success(f"Connected to Neo4j! Database has {result[0]['count']} nodes.")
        return graph
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}", exc_info=True)
        st.error(f"Neo4j Connection Error: {e}. Please check credentials and URI.")
        return None

def setup_neo4j_schema(graph):
    """Define and setup the graph schema"""
    logger.info("Setting up Neo4j schema and constraints")
    st.info("Setting up Neo4j schema (constraints and indexes)...")
    try:
        # Constraints (use IF NOT EXISTS)
        constraints = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (cp:Concept) REQUIRE cp.id IS UNIQUE", # Fixed label typo
            # Constraint on relationship property might not be standard Cypher or necessary
            # "CREATE CONSTRAINT relationship_type IF NOT EXISTS FOR ()-[r:MENTIONS]-() REQUIRE r.type IS NOT NULL",
        ]
        for constraint in constraints:
            try:
                graph.query(constraint)
                logger.debug(f"Applied constraint: {constraint}")
            except Exception as e:
                # Ignore errors if constraint already exists
                if "already exists" in str(e).lower():
                     logger.warning(f"Constraint likely already exists: {constraint}")
                else:
                    logger.error(f"Failed to apply constraint {constraint}: {e}", exc_info=True)
                    st.warning(f"Could not apply constraint: {constraint.split(' ')[2]}...") # Show affected label/property


        # Indexes (use IF NOT EXISTS)
        indexes = [
            "CREATE INDEX document_source_idx IF NOT EXISTS FOR (d:Document) ON (d.source)",
            "CREATE INDEX chunk_content_idx IF NOT EXISTS FOR (c:Chunk) ON (c.text)", # Use 'text' if using default from Neo4jVector
            "CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) OPTIONS { indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}", # Explicit Vector Index for MiniLM-L6-v2 (384 dimensions)
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX concept_name_idx IF NOT EXISTS FOR (cp:Concept) ON (cp.name)", # Fixed label typo
        ]
        for index in indexes:
             try:
                graph.query(index)
                logger.debug(f"Applied index: {index}")
             except Exception as e:
                if "already exists" in str(e).lower():
                    logger.warning(f"Index likely already exists: {index}")
                elif "supported for node properties" in str(e).lower() and "VECTOR INDEX" in index:
                    logger.warning(f"Vector index creation skipped or failed. Ensure Neo4j version supports vector indexing and has APOC: {index}")
                    st.warning("Vector index setup might require specific Neo4j version/plugins.")
                else:
                    logger.error(f"Failed to apply index {index}: {e}", exc_info=True)
                    st.warning(f"Could not apply index: {index.split(' ')[2]}...")

        logger.info("Neo4j schema setup attempt completed")
        st.success("Neo4j schema setup finished.")
    except Exception as e:
        logger.error(f"Error setting up Neo4j schema: {str(e)}", exc_info=True)
        st.error(f"Error setting up Neo4j schema: {e}")
        raise # Re-raise to stop processing if schema fails critically


# Define a helper function for concurrent entity extraction
def extract_entities_task(chunk, api_key, model_name, base_url):
    """
    Task function to extract entities for a single chunk using a temporary LLM client.
    Handles API calls and basic JSON parsing.
    """
    chunk_id = chunk.metadata['chunk_id']
    logger.debug(f"Starting entity extraction task for chunk {chunk_id}")
    try:
        # Create a *new*, temporary LLM instance for this task
        # Note: This instance won't share history with others or the main Q&A LLM
        temp_llm_client = DeepSeekLLM(
            api_key=api_key,
            model=model_name, # Use the reasoner or another suitable model
            max_tokens=256,
            temperature=0.1,
            api_base_url=base_url,
            messages=[] # Ensure it starts with empty history
        )

        entity_prompt = f"""
        Extract key named entities (Person, Organization, Location, Product, Technology) and important concepts/topics from the text below.
        Return them as a JSON list of objects, each with "name" (string) and "type" (string, e.g., "Person", "Concept", "Technology").
        Focus on relevance and significance within the context. Limit to the most important 5-7 items if many exist.

        TEXT:
        {chunk.page_content[:1500]} # Limit context size

        JSON RESPONSE (list of objects):
        """
        logger.info(f"Attempting LLM entity extraction for chunk {chunk_id}. Prompt length: {len(entity_prompt)}")
        entity_response = temp_llm_client.invoke(entity_prompt) # Uses the _call method with timeout
        logger.info(f"LLM entity extraction successful for chunk {chunk_id}. Response length: {len(entity_response)}")

        # Parse JSON robustly (same logic as before)
        entities = []
        try:
            json_match = re.search(r'\[.*?\]', entity_response, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group(0))
            else:
                json_match = re.search(r'\{.*?\}', entity_response, re.DOTALL)
                if json_match:
                     try:
                         potential_json = json_match.group(0)
                         if potential_json.count('{') > 1 and potential_json.count('}') > 1:
                              entities = json.loads(f"[{potential_json}]")
                         else:
                              entities = [json.loads(potential_json)]
                     except json.JSONDecodeError:
                          logger.warning(f"Could not decode JSON fragment in entity response for chunk {chunk_id}. Response: {entity_response[:200]}...")
                          entities = []
                else:
                    logger.warning(f"No JSON list or object found in entity response for chunk {chunk_id}. Response: {entity_response[:200]}...")
                    entities = []
        except json.JSONDecodeError as json_e:
            logger.warning(f"Failed to parse entity JSON for chunk {chunk_id}: {json_e}. Response: {entity_response[:200]}...")
            entities = []
        except Exception as parse_e:
            logger.warning(f"Unexpected error parsing entity JSON for chunk {chunk_id}: {parse_e}. Response: {entity_response[:200]}...")
            entities = []

        return chunk_id, entities

    except Exception as e:
        logger.error(f"Entity extraction task failed for chunk {chunk_id}: {e}", exc_info=False) # Keep log concise
        # Return the error to be handled later
        return chunk_id, e # Return chunk_id and the exception object


def create_knowledge_graph(graph, chunks, source_metadata, api_key):
    """Create a knowledge graph from the document chunks using concurrent entity extraction."""
    logger.info("Starting knowledge graph creation with concurrent processing.")
    st.info("Building knowledge graph (concurrent)... This can take a while.")
    total_chunks = len(chunks)
    progress_bar = st.progress(0, text="Initializing graph build...")
    processed_chunks = 0

    # Set the maximum number of concurrent workers (tune carefully based on API limits/performance)
    MAX_WORKERS = 5 # Start with a conservative number
    logger.info(f"Using up to {MAX_WORKERS} concurrent workers for entity extraction.")

    # Store base LLM parameters
    base_model_name = "deepseek-reasoner" # Or choose model for extraction
    base_api_url = "https://api.deepseek.com/v1"

    # --- Step 1: Create Chunk Nodes (Sequentially first) ---
    st.info("Creating base chunk nodes in the graph...")
    logger.info("Creating all chunk nodes sequentially first.")
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.metadata['chunk_id']
        source_doc = chunk.metadata['source_document']
        page_num = chunk.metadata.get('page', 0)
        try:
            query = """
            MATCH (d:Document {id: $doc_id})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text,
                c.page_num = $page_num,
                c.source_document = $doc_id
            MERGE (d)-[:CONTAINS]->(c)
            """
            params = {
                "chunk_id": chunk_id,
                "text": chunk.page_content,
                "page_num": page_num,
                "doc_id": source_doc
            }
            graph.query(query, params=params)
        except Exception as e:
             logger.error(f"Error creating chunk node {chunk_id} or linking: {e}", exc_info=True)
             st.warning(f"Skipping chunk node creation for {chunk_id} due to error. This may affect subsequent steps.")
             # Decide if you want to skip this chunk entirely for entity extraction
             # For now, we let it proceed, but entity extraction might fail if node doesn't exist
        progress_bar.progress((i + 1) / (total_chunks * 2), text=f"Creating chunk nodes {i+1}/{total_chunks}") # Adjust progress denominator

    # --- Step 2: Concurrent Entity Extraction ---
    logger.info("Starting concurrent entity extraction.")
    st.info(f"Extracting entities using up to {MAX_WORKERS} concurrent workers...")
    results = {} # Dictionary to store results: {chunk_id: entities_or_error}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(extract_entities_task, chunk, api_key, base_model_name, base_api_url): chunk
            for chunk in chunks
        }

        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            chunk_id = chunk.metadata['chunk_id']
            try:
                c_id, result_data = future.result()
                results[c_id] = result_data # Store entities list or Exception object
                if isinstance(result_data, Exception):
                    st.warning(f"Entity extraction failed for chunk {c_id}: {result_data}")
                # else:
                #     logger.debug(f"Successfully extracted {len(result_data)} entities for chunk {c_id}")

            except Exception as exc:
                logger.error(f'Chunk {chunk_id} generated an exception during future processing: {exc}', exc_info=True)
                results[chunk_id] = exc # Store the exception
                st.error(f"Error processing result for chunk {chunk_id}: {exc}")

            processed_chunks += 1
            progress_bar.progress(0.5 + (processed_chunks / (total_chunks * 2)), text=f"Extracting entities {processed_chunks}/{total_chunks}") # Adjust progress

    # --- Step 3: Create Entity Nodes and Relationships (Sequentially from results) ---
    logger.info("Processing entity extraction results and updating graph.")
    st.info("Adding extracted entities and relationships to the graph...")
    entities_added_count = 0
    processed_entities_step = 0

    for chunk_id, extracted_data in results.items():
        processed_entities_step += 1
        progress_bar.progress(0.75 + (processed_entities_step / (total_chunks * 4)), text=f"Adding entities to graph {processed_entities_step}/{total_chunks}") # Adjust progress

        if isinstance(extracted_data, Exception) or not extracted_data:
            # Skip chunks where extraction failed or returned no entities
            if isinstance(extracted_data, Exception):
                 logger.warning(f"Skipping graph update for chunk {chunk_id} due to previous extraction error: {extracted_data}")
            # else: logger.debug(f"No entities to add for chunk {chunk_id}")
            continue

        # 'extracted_data' should be the list of entities here
        entities = extracted_data
        for entity in entities:
             if not isinstance(entity, dict) or 'name' not in entity or 'type' not in entity:
                 logger.warning(f"Invalid entity format skipped: {entity} in chunk {chunk_id}")
                 continue
             entity_name = str(entity.get('name')).strip()
             entity_type = str(entity.get('type')).strip().capitalize()
             if not entity_name or not entity_type: continue # Skip if invalid

             clean_name_part = re.sub(r'\s+', '_', re.sub(r'[^\w\s-]', '', entity_name).strip()).lower()
             clean_type_part = re.sub(r'\s+', '_', entity_type).lower()
             entity_id = f"{clean_name_part}_{clean_type_part}"[:255]
             node_label = "Concept" if entity_type.lower() in ["concept", "topic"] else "Entity"

             try:
                 logger.debug(f"Creating {node_label} node {entity_id} ({entity_name}) and linking to chunk {chunk_id}")
                 query = f"""
                 MERGE (e:{node_label} {{id: $entity_id}})
                 ON CREATE SET e.name = $name, e.type = $type
                 WITH e
                 MATCH (c:Chunk {{id: $chunk_id}})
                 MERGE (c)-[r:MENTIONS {{type: $type}}]->(e)
                 """
                 params = { "entity_id": entity_id, "name": entity_name, "type": entity_type, "chunk_id": chunk_id }
                 graph.query(query, params=params)
                 entities_added_count += 1
             except Exception as e:
                  logger.error(f"Error creating {node_label} node {entity_id} or relationship: {e}", exc_info=True)
                  st.warning(f"Failed to create/link entity {entity_name} from chunk {chunk_id}.")
                  continue # Continue with the next entity

    logger.info(f"Added {entities_added_count} entity mentions to the graph.")

    # --- Step 4: Link Related Entities ---
    progress_bar.progress(0.95, text="Linking related entities...")
    logger.info("Creating connections between related entities within the same chunk context...")
    try:
        query = """
        MATCH (c:Chunk)-[:MENTIONS]->(e1)
        MATCH (c)-[:MENTIONS]->(e2)
        WHERE id(e1) < id(e2)
        MERGE (e1)-[r:RELATED_TO]->(e2)
        ON CREATE SET r.weight = 1
        ON MATCH SET r.weight = r.weight + 1
        """
        graph.query(query)
        logger.info("Finished linking related entities.")
    except Exception as e:
        logger.error(f"Failed to link related entities: {e}", exc_info=True)
        st.error(f"Error during final relationship linking: {e}")
        # Decide if this error is critical enough to return False

    progress_bar.progress(1.0, text="Knowledge graph structure build completed!")
    st.success("Knowledge graph structure built successfully!")
    progress_bar.empty()
    return True # Indicate success

def setup_vector_index(chunks, embedding_function, neo4j_uri, neo4j_username, neo4j_password):
    """Setup vector embeddings in Neo4j for semantic search"""
    logger.info("Setting up vector index for semantic search")
    st.info("Creating vector embeddings and indexing in Neo4j...")
    if not chunks:
        st.warning("No chunks available to create vector index.")
        return None

    try:
        # Ensure embedding function is loaded
        if embedding_function is None:
            st.error("Embedding function not available. Cannot create vector index.")
            return None

        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks] # Pass metadata for linking

        # Use Neo4jVector.from_documents for better metadata handling
        # It assumes nodes (Chunks in this case) already exist and adds embeddings
        # It requires node_label, text_node_property, and embedding_node_property
        # It will match existing nodes based on a property (e.g., 'id') if specified or create new ones
        # Let's ensure our graph creation step created the :Chunk nodes with 'id' and 'text'.

        logger.info(f"Attempting to add/update {len(texts)} chunk embeddings to Neo4jVector...")

        vector_index = Neo4jVector.from_texts(
             texts=texts,
             embedding=embedding_function,
             metadatas=metadatas, # Include metadata
             url=neo4j_uri,
             username=neo4j_username,
             password=neo4j_password,
             index_name="chunk_embeddings", # Name of the vector index in Neo4j
             node_label="Chunk",            # Label of nodes to store embeddings
             text_node_property="text",     # Property containing the text (matches our graph creation)
             embedding_node_property="embedding", # Property to store the embedding vector
             # IMPORTANT: Define how to link/match existing nodes. Match by 'id'.
             # This requires slightly different setup or assumes from_texts creates if not exists.
             # For robustness, let's assume `from_texts` handles MERGE-like behavior based on index/constraints.
             # If issues arise, consider `from_documents` with pre-fetched nodes or custom Cypher.
        )


        # Neo4jVector.from_texts usually handles index creation or verification.
        # The explicit index creation in setup_neo4j_schema is a good safeguard.

        logger.info(f"Vector index setup completed for {len(chunks)} chunks.")
        st.success(f"Vector index ready with {len(chunks)} chunk embeddings.")
        return vector_index

    except Exception as e:
        logger.error(f"Error setting up vector index: {str(e)}", exc_info=True)
        st.error(f"Error during vector index setup: {e}")
        # Try to provide more specific feedback if possible
        if "authentication" in str(e).lower():
             st.error("Check Neo4j connection details.")
        elif "vector index" in str(e).lower():
             st.error("Ensure Neo4j version/plugins support vector indexing.")
        return None


# Query Functions
def extract_entities_from_question(question):
    """Simple keyword/noun phrase extraction for entity matching"""
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'through', 'over', 'before', 'after', 'since', 'of', 'from', 'what', 'who', 'where', 'when', 'how', 'is', 'are', 'was', 'were', 'tell', 'me', 'show'}
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', question.lower()) # Allow hyphens/underscores
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

    # Simple Noun Phrase (Adjective/Noun + Noun) - less crucial than keywords here
    # text = question.lower()
    # phrases = re.findall(r'\b[a-z]+\s+[a-z]+\b', text)
    # phrases = [p for p in phrases if not any(word in stop_words for word in p.split())]

    # Prioritize proper nouns (capitalized words not at sentence start - simplistic)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', question)
    proper_nouns = [pn.lower() for pn in proper_nouns if pn.lower() not in stop_words]

    entities = list(set(filtered_words + proper_nouns))
    logger.debug(f"Extracted potential entities from question: {entities}")
    return entities

def get_query_context(question, vector_store, graph):
    """Get context for query using vector similarity and graph traversal"""
    logger.info(f"Getting context for question: {question}")
    st.info("Retrieving relevant information...")
    contexts = []
    retrieved_chunk_ids = set() # Keep track of retrieved chunks to avoid duplicates

    try:
        # Step 1: Vector Similarity Search
        if vector_store:
            logger.debug("Performing vector similarity search")
            try:
                vector_results = vector_store.similarity_search(question, k=3) # Retrieve top 3 vector matches
                logger.debug(f"Vector search found {len(vector_results)} relevant chunks")
                for idx, doc in enumerate(vector_results):
                    chunk_id = doc.metadata.get('chunk_id', 'vector_unknown')
                    if chunk_id not in retrieved_chunk_ids:
                        contexts.append({
                            "source": "Vector Similarity",
                            "rank": idx,
                            "chunk_id": chunk_id,
                            "content": doc.page_content,
                            "source_document": doc.metadata.get('source_document', 'unknown'),
                            "page_num": doc.metadata.get('page', '?')
                        })
                        retrieved_chunk_ids.add(chunk_id)
            except Exception as e:
                 logger.error(f"Vector similarity search failed: {e}", exc_info=True)
                 st.warning("Could not perform vector similarity search.")
        else:
             logger.warning("Vector store not available for context retrieval.")
             st.warning("Vector index not available, skipping vector search.")


        # Step 2: Graph Traversal based on Entities in Question
        logger.debug("Identifying key entities in the question for graph traversal")
        entities_in_question = extract_entities_from_question(question)

        if entities_in_question and graph:
            logger.debug(f"Performing graph traversal for entities: {entities_in_question}")
            max_graph_results_per_entity = 2 # Limit results per entity to avoid flooding context

            for entity in entities_in_question:
                 entity_context_count = 0 # Track results for this entity
                 # Find chunks directly mentioning the entity (or similar name)
                 try:
                      query_direct = """
                      MATCH (c:Chunk)-[:MENTIONS]->(e) // Entity or Concept
                      WHERE (e.name IS NOT NULL AND toLower(e.name) CONTAINS toLower($entity_name))
                         OR (e.id IS NOT NULL AND toLower(e.id) CONTAINS toLower($entity_name)) // Match ID too
                      RETURN c.id as chunk_id, c.text as content, c.page_num as page_num, e.name as entity_name, labels(e)[0] as entity_type
                      LIMIT $limit
                      """
                      graph_results_direct = graph.query(query_direct, params={"entity_name": entity, "limit": max_graph_results_per_entity})

                      for idx, result in enumerate(graph_results_direct):
                          chunk_id = result.get('chunk_id', f'graph_{entity}_direct_{idx}')
                          if chunk_id not in retrieved_chunk_ids:
                              contexts.append({
                                  "source": "Graph: Direct Mention",
                                  "entity": result.get('entity_name', entity),
                                  "rank": idx,
                                  "chunk_id": chunk_id,
                                  "content": result.get('content', ''),
                                  "page_num": result.get('page_num', '?'),
                                  "entity_type": result.get('entity_type', 'Unknown')
                              })
                              retrieved_chunk_ids.add(chunk_id)
                              entity_context_count += 1
                              if entity_context_count >= max_graph_results_per_entity: break # Stop if limit reached

                      # Find chunks mentioning related entities (1 hop) - only if needed
                      if entity_context_count < max_graph_results_per_entity:
                          query_related = """
                          MATCH (e1)-[:RELATED_TO]-(e2) // Entity or Concept, relationship can be either direction
                          WHERE (e1.name IS NOT NULL AND toLower(e1.name) CONTAINS toLower($entity_name))
                             OR (e1.id IS NOT NULL AND toLower(e1.id) CONTAINS toLower($entity_name))
                          MATCH (c:Chunk)-[:MENTIONS]->(e2) // Find chunks mentioning the related entity
                          WHERE c.id IS NOT NULL
                          RETURN c.id as chunk_id, c.text as content, c.page_num as page_num,
                                 e1.name as source_entity, e2.name as related_entity, labels(e2)[0] as related_type
                          LIMIT $limit
                          """
                          # Calculate remaining limit
                          remaining_limit = max_graph_results_per_entity - entity_context_count
                          graph_results_related = graph.query(query_related, params={"entity_name": entity, "limit": remaining_limit})

                          for idx, result in enumerate(graph_results_related):
                              chunk_id = result.get('chunk_id', f'graph_{entity}_related_{idx}')
                              if chunk_id not in retrieved_chunk_ids:
                                  contexts.append({
                                      "source": "Graph: Related Mention",
                                      "entity": entity, # The entity from the question
                                      "related_entity": result.get('related_entity', 'unknown'),
                                      "rank": idx,
                                      "chunk_id": chunk_id,
                                      "content": result.get('content', ''),
                                      "page_num": result.get('page_num', '?'),
                                      "related_type": result.get('related_type', 'Unknown')
                                  })
                                  retrieved_chunk_ids.add(chunk_id)
                                  entity_context_count += 1
                                  if entity_context_count >= max_graph_results_per_entity: break # Stop if limit reached

                 except Exception as e:
                      logger.error(f"Graph traversal for entity '{entity}' failed: {e}", exc_info=True)
                      st.warning(f"Could not perform graph search for entity: {entity}")

        # Removed the specific Connector-Attribute schema queries as they weren't well-defined
        # The generic Entity/Concept traversal should cover relevant info if extracted correctly.

        # De-duplicate contexts based on chunk_id (though already handled by retrieved_chunk_ids set)
        final_contexts = []
        seen_ids = set()
        for ctx in contexts:
            ctx_id = ctx.get('chunk_id', None)
            if ctx_id and ctx_id not in seen_ids:
                final_contexts.append(ctx)
                seen_ids.add(ctx_id)
            elif not ctx_id: # Keep contexts without chunk_id if necessary (e.g., pure graph relation data)
                final_contexts.append(ctx)


        logger.info(f"Context gathering completed. Found {len(final_contexts)} unique relevant contexts from sources: {list(set([c['source'] for c in final_contexts]))}")
        st.info(f"Retrieved {len(final_contexts)} context snippets.")
        return final_contexts

    except Exception as e:
        logger.error(f"Error getting query context: {str(e)}", exc_info=True)
        st.error(f"Error retrieving context: {e}")
        return contexts # Return potentially partial contexts


def answer_question(question, api_key, contexts, neo4j_uri, neo4j_username, neo4j_password):
    """Generate answer based on retrieved contexts using DeepSeek API"""
    logger.info(f"Generating answer for question: {question}")
    st.info("Synthesizing answer using retrieved context...")
    start_time = time.time()

    if not api_key:
        st.error("DeepSeek API Key is missing. Cannot generate answer.")
        return {"error": "DeepSeek API Key is missing."}

    if not contexts:
         st.warning("No context was retrieved. Attempting to answer based on general knowledge (may be inaccurate).")
         # Optionally, provide a direct prompt without context or return a specific message
         # return {"answer": "I couldn't find specific information in the documents to answer that. Please try rephrasing or asking about the content of the uploaded PDFs.", "processing_time": 0, "contexts_used": 0, "context_sources": []}


    try:
        # Compile context into a readable format for the LLM
        context_text = ""
        if contexts:
            context_text += "Relevant information found:\n"
            for i, ctx in enumerate(contexts[:5]): # Limit context injection to top 5-7 to manage token limits
                 context_text += f"\n--- Context Snippet {i+1} ---\n"
                 context_text += f"Source Type: {ctx.get('source', 'Unknown')}\n"
                 if 'chunk_id' in ctx:
                     context_text += f"Document: {ctx.get('source_document', 'Unknown')}, Page: {ctx.get('page_num', '?')}, Chunk ID: {ctx.get('chunk_id', 'N/A')}\n"
                 if 'entity' in ctx:
                     context_text += f"Related Entity: {ctx.get('entity', 'N/A')}\n"
                 if 'related_entity' in ctx:
                     context_text += f"Mentioned Entity: {ctx.get('related_entity', 'N/A')} ({ctx.get('related_type', 'N/A')})\n"

                 # Limit content length per snippet
                 content_preview = ctx.get('content', '')[:500] # Limit snippet length
                 context_text += f"Content: {content_preview}...\n"
            context_text += "\n---\n"


        # Add graph schema information (optional, can make prompt long)
        # graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
        # try:
        #     schema_text = graph.get_schema # This can be verbose
        #     logger.debug(f"Neo4j Schema: {schema_text}")
        # except Exception as e:
        #      logger.warning(f"Could not retrieve Neo4j schema: {e}")
        #      schema_text = "Node Labels: Document, Chunk, Entity, Concept. Relationships: CONTAINS, MENTIONS, RELATED_TO."
        schema_text = "Knowledge Representation: Documents contain Chunks. Chunks mention Entities/Concepts. Entities/Concepts can be RELATED_TO each other."


        # Formulate the prompt
        prompt = f"""You are an AI assistant answering questions based *only* on the provided context information extracted from documents and a knowledge graph.
If the context does not contain the answer, state that clearly. Do not make up information.
Be concise and directly answer the question. Reference the source document or page number if available in the context.

Knowledge Graph Structure: {schema_text}

{context_text}

Based *only* on the context above, answer the following question:
Question: {question}

Answer:"""

        # LLM Call - Use session state to maintain conversation history
        if 'llm' not in st.session_state:
            st.session_state.llm = DeepSeekLLM(api_key=api_key)
        
        # Get the answer
        answer_raw = st.session_state.llm.invoke(prompt)

        # Basic cleaning
        cleaned_answer = answer_raw.strip()

        end_time = time.time()
        result = {
             "answer": cleaned_answer,
             "processing_time": end_time - start_time,
             "contexts_used": len(contexts),
             "context_sources": list(set([c.get('source', 'Unknown') for c in contexts]))
        }
        logger.info(f"Answer generated in {result['processing_time']:.2f} seconds.")
        st.success("Answer generated.")
        return result

    except Exception as e:
        logger.error(f"Error generating answer with LLM: {str(e)}", exc_info=True)
        st.error(f"Error generating answer: {e}")
        return {"error": f"Failed to generate answer: {str(e)}"}

# =============================
# Streamlit UI Setup
# =============================

st.set_page_config(layout="wide", page_title="Graph RAG Assistant")

st.title("📄🔗 Graph RAG Document Assistant")
st.markdown("Upload PDFs, build a knowledge graph, and ask questions about the content.")

# Initialize session state variables
if 'neo4j_uri' not in st.session_state:
    st.session_state.neo4j_uri = ""
if 'neo4j_username' not in st.session_state:
    st.session_state.neo4j_username = ""
if 'neo4j_password' not in st.session_state:
    st.session_state.neo4j_password = ""
if 'deepseek_api_key' not in st.session_state:
    st.session_state.deepseek_api_key = ""
if 'neo4j_connected' not in st.session_state:
    st.session_state.neo4j_connected = False
if 'graph_built' not in st.session_state:
    st.session_state.graph_built = False
if 'graph_data' not in st.session_state:
    # Stores Neo4jGraph object, vector_store, stats, etc. after processing
    st.session_state.graph_data = None
if 'uploaded_files_processed' not in st.session_state:
    # Track if the current set of uploaded files has been processed
    st.session_state.uploaded_files_processed = False
if 'current_uploaded_files' not in st.session_state:
    # Store the names/ids of currently uploaded files
    st.session_state.current_uploaded_files = []


# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")

    # Neo4j Credentials
    st.subheader("Neo4j Connection")
    # Use st.secrets if available, otherwise show input fields
    default_uri = st.secrets.get("NEO4J_URI", "")
    default_user = st.secrets.get("NEO4J_USERNAME", "neo4j") # Default user often 'neo4j'
    default_pass = st.secrets.get("NEO4J_PASSWORD", "")

    st.session_state.neo4j_uri = st.text_input("Neo4j URI", value=st.session_state.neo4j_uri or default_uri, placeholder="bolt://<host>:<port> or neo4j+s://<...> (Aura)")
    st.session_state.neo4j_username = st.text_input("Neo4j Username", value=st.session_state.neo4j_username or default_user)
    st.session_state.neo4j_password = st.text_input("Neo4j Password", type="password", value=st.session_state.neo4j_password or default_pass)

    if st.button("Connect to Neo4j"):
        if st.session_state.neo4j_uri and st.session_state.neo4j_username and st.session_state.neo4j_password:
            with st.spinner("Connecting to Neo4j..."):
                graph = connect_to_neo4j(
                    st.session_state.neo4j_uri,
                    st.session_state.neo4j_username,
                    st.session_state.neo4j_password
                )
                if graph:
                    st.session_state.neo4j_connected = True
                    # Store graph object temporarily if needed, but graph_data will hold it after processing
                    # st.session_state.temp_graph_obj = graph
                else:
                    st.session_state.neo4j_connected = False
        else:
            st.warning("Please provide all Neo4j connection details.")


    if st.session_state.neo4j_connected:
        st.success("✅ Neo4j Connected")
    else:
        st.info("Provide Neo4j credentials and click 'Connect'.")


    # DeepSeek API Key
    st.subheader("DeepSeek API Key")
    # Check st.secrets for the key, otherwise use session state or show empty input
    default_api_key = st.secrets.get("DEEPSEEK_API_KEY", "") # Use a specific secret if desired
    st.session_state.deepseek_api_key = st.text_input(
        "DeepSeek API Key", # Changed label
        type="password",
        value=st.session_state.deepseek_api_key or default_api_key, # Use renamed key
        help="Get from DeepSeek platform." # Changed help text
    )

    # Check if the key looks plausible (basic check)
    if st.session_state.deepseek_api_key: # Simple check if key exists
        st.success("✅ DeepSeek Key Provided")
    else:
        st.info("Provide your DeepSeek API key.")


    # --- File Upload ---
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        accept_multiple_files=True,
        type="pdf",
        key="pdf_uploader" # Use key to help detect changes
    )

    # Detect if new files have been uploaded
    current_file_names = sorted([f.name for f in uploaded_files]) if uploaded_files else []
    if current_file_names != st.session_state.current_uploaded_files:
        st.session_state.uploaded_files_processed = False # Reset processed flag if files change
        st.session_state.graph_built = False             # Reset graph built flag
        st.session_state.graph_data = None              # Clear old graph data
        st.session_state.current_uploaded_files = current_file_names
        st.info("New files detected. Ready to process.")

# --- Main Area ---

# Step 1: Process PDFs and Build Graph
st.header("1. Process PDFs & Build Knowledge Graph")

if not st.session_state.neo4j_connected:
    st.warning("⚠️ Please connect to Neo4j in the sidebar first.")
elif not st.session_state.deepseek_api_key:
     st.warning("⚠️ Please provide your DeepSeek API key in the sidebar.")
elif not uploaded_files:
    st.info("⬆️ Upload one or more PDF files using the sidebar.")
elif st.session_state.uploaded_files_processed:
     st.success(f"✅ Files ({len(st.session_state.current_uploaded_files)}) already processed. Graph is ready.")
     st.write(f"**Documents:** {st.session_state.graph_data.get('doc_count', 'N/A')}, "
              f"**Chunks:** {st.session_state.graph_data.get('chunk_count', 'N/A')}, "
              f"**Entities/Concepts:** {st.session_state.graph_data.get('entity_count', 'N/A')}")

else:
    if st.button("Process Files & Build Graph"):
        st.session_state.graph_built = False # Reset flag before starting
        st.session_state.graph_data = None

        with st.spinner("Processing PDFs, building graph, and creating embeddings... This may take some time."):
            # Get embedding function (cached)
            embedding_function = get_embeddings()
            if not embedding_function:
                st.error("Failed to load embedding model. Cannot proceed.")
                st.stop()

            # Connect to Neo4j (re-connect here to ensure freshness)
            graph = connect_to_neo4j(st.session_state.neo4j_uri, st.session_state.neo4j_username, st.session_state.neo4j_password)
            if not graph:
                st.error("Neo4j connection failed. Cannot proceed.")
                st.stop()

            # Process PDFs
            chunks, source_metadata = process_pdfs_streamlit(uploaded_files)
            if not chunks:
                st.error("No content extracted from PDFs. Stopping.")
                st.stop()

            # Setup Schema
            try:
                setup_neo4j_schema(graph)
            except Exception as schema_e:
                st.error(f"Failed during schema setup: {schema_e}. Stopping.")
                st.stop()


            # Create Knowledge Graph structure
            graph_creation_success = create_knowledge_graph(graph, chunks, source_metadata, st.session_state.deepseek_api_key)
            if not graph_creation_success:
                 st.error("Failed during knowledge graph structure creation. Check logs. Stopping.")
                 st.stop()

            # Setup Vector Index
            vector_store = setup_vector_index(chunks, embedding_function, st.session_state.neo4j_uri, st.session_state.neo4j_username, st.session_state.neo4j_password)
            if not vector_store:
                 st.error("Failed during vector index setup. QA might be impaired. Stopping.")
                 # Decide if you want to continue without vector search or stop
                 st.stop()


            # Get updated graph stats
            try:
                doc_count = graph.query("MATCH (d:Document) RETURN count(d) as count")[0]['count']
                chunk_count = graph.query("MATCH (c:Chunk) RETURN count(c) as count")[0]['count']
                entity_count = graph.query("MATCH (e) WHERE e:Entity OR e:Concept RETURN count(e) as count")[0]['count']
            except Exception as stat_e:
                 logger.warning(f"Could not retrieve final graph stats: {stat_e}")
                 doc_count, chunk_count, entity_count = 'N/A', 'N/A', 'N/A'


            # Store results in session state
            st.session_state.graph_data = {
                "graph": graph, # Store the connected graph object
                "vector_store": vector_store, # Store the vector store object
                "neo4j_uri": st.session_state.neo4j_uri, # Keep connection info
                "neo4j_username": st.session_state.neo4j_username,
                "neo4j_password": st.session_state.neo4j_password, # Be mindful if this needs to be persisted securely
                "doc_count": doc_count,
                "chunk_count": chunk_count,
                "entity_count": entity_count,
                "embedding_function": embedding_function # Include if needed later
            }
            st.session_state.graph_built = True
            st.session_state.uploaded_files_processed = True # Mark current files as processed

        st.success("✅ PDF Processing, Graph Building, and Vector Indexing Complete!")
        st.write(f"**Documents:** {doc_count}, **Chunks:** {chunk_count}, **Entities/Concepts:** {entity_count}")


# Step 2: Ask Questions
st.header("2. Ask Questions")

if not st.session_state.graph_built or not st.session_state.graph_data:
    st.info("☝️ Please process PDF files first to enable the Q&A section.")
else:
    # Add clear history button
    if st.button("Clear Conversation History"):
        if 'llm' in st.session_state:
            st.session_state.llm.clear_history()
            st.success("Conversation history cleared!")

    question = st.text_area("Enter your question about the documents:", height=100)

    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question.")
        elif not st.session_state.deepseek_api_key:
             st.error("DeepSeek API Key missing. Cannot generate answer.")
        else:
            with st.spinner("Searching knowledge graph and generating answer..."):
                # Retrieve necessary components from session state
                graph_data = st.session_state.graph_data
                api_key = st.session_state.deepseek_api_key

                # Get context
                contexts = get_query_context(
                    question,
                    graph_data["vector_store"],
                    graph_data["graph"]
                )

                # Generate answer
                result = answer_question(
                    question,
                    api_key,
                    contexts,
                    graph_data["neo4j_uri"],
                    graph_data["neo4j_username"],
                    graph_data["neo4j_password"]
                )

                # Display answer
                st.subheader("Answer:")
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.markdown(result["answer"])
                    st.divider()
                    # Show context details in an expander
                    with st.expander("View Context Used"):
                        st.write(f"Processing time: {result['processing_time']:.2f} seconds")
                        st.write(f"Contexts retrieved: {result['contexts_used']}")
                        st.write(f"Context sources: {', '.join(result['context_sources'])}")
                        st.json(contexts) # Display raw context for debugging


# Step 3: Debugging Tools (Optional)
st.header("3. Debugging & Analysis")

if not st.session_state.graph_built or not st.session_state.graph_data:
    st.info("☝️ Process PDF files first to enable debugging tools.")
else:
    debug_tab1, debug_tab2 = st.tabs(["Graph Stats", "Query Analysis"])

    with debug_tab1:
        st.subheader("Knowledge Graph Statistics")
        if st.button("Show Graph Stats"):
             with st.spinner("Fetching graph statistics..."):
                graph = st.session_state.graph_data["graph"]
                try:
                    doc_query = "MATCH (d:Document) RETURN d.id as document, d.pages as pages, count{(d)-[:CONTAINS]->(:Chunk)} as chunks ORDER BY document"
                    doc_stats = graph.query(doc_query)
                    st.write("**Documents & Chunks:**")
                    st.dataframe(pd.DataFrame(doc_stats), use_container_width=True)

                    entity_query = "MATCH (e) WHERE e:Entity OR e:Concept WITH labels(e)[0] as label, e.type as type, count(e) as count RETURN label, type, count ORDER BY count DESC LIMIT 20"
                    entity_stats = graph.query(entity_query)
                    st.write("**Top Entity/Concept Types:**")
                    st.dataframe(pd.DataFrame(entity_stats), use_container_width=True)

                    rel_query = "MATCH (e1)-[r:RELATED_TO]->(e2) RETURN labels(e1)[0] + ' ' + e1.name as source, labels(e2)[0] + ' ' + e2.name as target, r.weight as weight ORDER BY r.weight DESC LIMIT 20"
                    rel_stats = graph.query(rel_query)
                    st.write("**Top Entity Relationships (by co-occurrence):**")
                    st.dataframe(pd.DataFrame(rel_stats), use_container_width=True)

                except Exception as e:
                    st.error(f"Failed to fetch graph stats: {e}")


    with debug_tab2:
        st.subheader("Analyze Query Context Retrieval")
        analyze_question = st.text_input("Enter a question to analyze context retrieval:")
        if st.button("Analyze Query"):
            if not analyze_question:
                st.warning("Please enter a question to analyze.")
            else:
                with st.spinner("Analyzing context retrieval steps..."):
                    graph_data = st.session_state.graph_data
                    contexts = get_query_context(
                        analyze_question,
                        graph_data["vector_store"],
                        graph_data["graph"]
                    )
                    st.write(f"**Analysis for question:** '{analyze_question}'")
                    st.write(f"**Total Contexts Retrieved:** {len(contexts)}")

                    st.write("**Context Breakdown:**")
                    if contexts:
                         # Use pandas for better display if many contexts
                         df_context = pd.DataFrame(contexts)
                         # Select and reorder columns for clarity
                         display_cols = ["source", "chunk_id", "entity", "related_entity", "page_num", "content"]
                         # Filter to existing columns only
                         existing_display_cols = [col for col in display_cols if col in df_context.columns]
                         st.dataframe(df_context[existing_display_cols], use_container_width=True)

                         # Display full content if needed
                         # for i, ctx in enumerate(contexts):
                         #      with st.expander(f"Context {i+1} ({ctx.get('source', '?')}) - ID: {ctx.get('chunk_id', 'N/A')}"):
                         #          st.json(ctx)

                    else:
                         st.write("No context snippets were retrieved for this question.")