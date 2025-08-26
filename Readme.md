
## 🤖 RAG Pipeline - Keyence Intelligent Document System

The **RAGpipeline.ipynb** implements a complete **Retrieval-Augmented Generation (RAG)** system specifically designed to process, index and query technical documentation of Keyence products.

### 🏗️ RAG Pipeline Architecture

```
📄 PDF Documents (Azure Blob) → 🔍 Document Intelligence → 🧠 GPT Structuring → 
📊 Hierarchical JSON → 🔢 Vector Embeddings → 🔍 Azure AI Search → 💬 Conversational AI
```

### 🔧 Main Components

#### 1. **Document Processing Layer**
- **Azure Document Intelligence**: Extracts text from PDFs stored in Azure Blob Storage
- **GPT-4 Structuring**: Normalizes and structures extracted content in hierarchical format
- **Hierarchical Segmentation**: Organizes information in A.1, A.2, B.1, B.2, etc. structure

#### 2. **Text Processing Functions**

| Function | Description |
|---------|-------------|
| `analyze_read(url)` | Extracts text from PDFs using Azure Document Intelligence |
| `normalize_with_GPT(text, query)` | Structures content with GPT-4 according to specific instructions |
| `segment_text_as_json(text)` | Converts structured text to hierarchical JSON format |
| `create_hierarchical_chunking(text)` | Parses text into nested dictionary structure |
| `string_to_json(json_string)` | Converts JSON strings to Python objects |

#### 3. **Vector Search Implementation**
- **Embedding Model**: `text-embedding-3-large` from Azure OpenAI
- **Vector Dimensions**: 3072 dimensions
- **Search Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Hybrid Search**: Combines vectorial and textual search

#### 4. **Azure AI Search Integration**
- **Index Management**: Automatic creation and management of indexes
- **Document Schema**:
  ```json
  {
    "id": "unique_uuid",
    "title": "Section title (A., B., C.)",
    "subtitle": "Subsection title (1., 2., 3.)",
    "content": "Detailed content text",
    "contentVector": [3072-dim embedding],
    "category": "Keyence",
    "additionalMetadata": "manual=source_url"
  }
  ```

#### 5. **Conversational AI System**
- **Sales Agent**: "Alberto" - Agent specialized in Keyence products
- **Context-Aware**: Uses retrieved information for precise responses
- **Response Limits**: Maximum 1600 characters per response
- **Follow-up Suggestions**: Suggests 2 relevant topics to continue the conversation

### 🔄 Detailed Workflow

#### **Phase 1: Document Ingestion**
```python
# 1. Connection to Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
blobs = [blob.name for blob in container_client.list_blobs()]

# 2. Processing each PDF
for blob_name in blobs:
    url = f"https://storage.blob.core.windows.net/files/{blob_name}"
    text = analyze_read(url)  # Azure Document Intelligence
```

#### **Phase 2: Intelligent Structuring**
```python
# 3. Structuring with GPT-4
query = '''1. Write the document in structured form dividing text hierarchically with bullets
           2. Use alphabet letters (A., B., C.) for topics, numbers (1., 2., 3.) for subtopics
           3. Develop all descriptions in detail'''
           
normalized_text = normalize_with_GPT(text, query)

# 4. Conversion to hierarchical JSON
chunks = segment_text_as_json(normalized_text)
```

#### **Phase 3: Vectorization and Indexing**
```python
# 5. Embedding generation
for section_title, subsections in data.items():
    for subsection_title, content_list in subsections.items():
        text_4_vector = section_title + " " + subsection_title
        vector = generate_embedding(client, text_4_vector, embedding_model)
        
        # 6. Creation of indexable document
        doc = {
            "id": str(uuid.uuid4()),
            "title": section_title,
            "subtitle": subsection_title,
            "content": content_text,
            "contentVector": vector,
            "category": "Keyence",
            "additionalMetadata": f"manual={url}"
        }
```

#### **Phase 4: Hybrid Search**
```python
# 7. Query processing
def chat_GPT(query):
    query_embedding = generate_embedding(azure_openai_client, query, EMBEDDING_MODEL)
    
    # 8. Hybrid search payload
    payload = {
        "search": query,  # Text search
        "vectorQueries": [{
            "kind": "vector",
            "vector": query_embedding,  # Vector search
            "fields": "contentVector",
            "k": 3
        }],
        "top": 10
    }
```

#### **Phase 5: Contextual Generation**
```python
# 9. Context building
results = response.json()['value']
context = " ".join([doc["category"] + ", " + doc["title"] + ", " + 
                   doc["subtitle"] + ", " + doc["content"] for doc in results])

# 10. GPT-4 response generation
system_prompt = f'''Your name is Alberto and you are an expert sales agent for Keyence.
                   Use context: {context}'''
```

### 📊 System Configuration

#### **Azure Services Required**
- **Azure Document Intelligence**: For PDF text extraction
- **Azure OpenAI**: For embeddings and response generation
- **Azure AI Search**: For indexing and hybrid search
- **Azure Blob Storage**: For document storage

#### **Models Used**
- **GPT-4**: For content structuring and response generation
- **text-embedding-3-large**: For content vectorization
- **Embedding Dimensions**: 3072

#### **Search Configuration**
```json
{
  "vectorSearch": {
    "algorithms": [{
      "name": "vector-config-19920811",
      "kind": "hnsw",
      "hnswParameters": {
        "metric": "cosine",
        "m": 4,
        "efConstruction": 400,
        "efSearch": 500
      }
    }]
  }
}
```

### 🎯 Use Cases

1. **Technical Queries**: "What are the specifications of the AP-N sensor?"
2. **Product Comparison**: "Differences between GL-R and GL-V Series"
3. **Specific Applications**: "Sensors for pressure detection in production lines"
4. **Installation Information**: "How to install safety curtains?"
5. **Compatibility**: "What protocols does the AP-N Series support?"

### 🔍 Advanced Features

- **Semantic Search**: Finds relevant information even when exact words don't match
- **Context Awareness**: Maintains conversation context for coherent responses
- **Metadata Enrichment**: Includes links to original manuals
- **Batch Processing**: Processes multiple documents efficiently
- **Error Handling**: Robust error handling in each phase of the pipeline

### 📈 Metrics and Optimization

- **Chunking Strategy**: Hierarchical segmentation for better retrieval
- **Vector Similarity**: Cosine search for maximum precision
- **Response Quality**: Responses limited to 1600 characters for better UX
- **Follow-up Generation**: Automatic suggestions to continue conversation

