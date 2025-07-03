# Simple LangChain RAG Pipeline – Data Loading, Transformation & Embeddings

This notebook demonstrates a **simple yet effective RAG (Retrieval-Augmented Generation) pipeline** using [LangChain](https://www.langchain.com/). It walks through the full lifecycle of:

1. Ingesting data from **local text**, **web pages**, and **PDFs**
2. Transforming documents using chunking
3. Creating embeddings using **Hugging Face**
4. Storing them in vector stores like **Chroma** and **FAISS**
5. Performing semantic search queries on the indexed data

---

## 📁 Pipeline Overview

### ✅ Step 1: Data Ingestion

We load data using different **LangChain loaders**:

- 📄 `TextLoader` – Load from `.txt` files (`speech.txt`)
- 🌐 `WebBaseLoader` – Load from a blog page (e.g., [Lilian Weng's blog](https://lilianweng.github.io/posts/2023-06-23-agent/)) with `BeautifulSoup` filtering
- 📚 `PyPDFLoader` – Load from PDF documents like `Colon.pdf`

### ✅ Step 2: Data Transformation

We split large text into manageable chunks using:

```python
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
````

This is essential for effective embedding and search performance.

### ✅ Step 3: Vector Embedding

We use Hugging Face embeddings via:

```python
HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

Other alternatives (commented out):

* `OpenAIEmbeddings`
* `OllamaEmbeddings`

### ✅ Step 4: Vector Stores

We tested two vector store implementations:

* 🧠 **Chroma**

```python
db1 = Chroma.from_documents(documents, embedding_model)
```

* 📦 **FAISS**

```python
db2 = FAISS.from_documents(documents[:30], embedding_model)
```

Both support similarity search based on vector embeddings.

### ✅ Step 5: Semantic Search (Querying)

Perform a similarity search over the stored vectors:

```python
query = "The choice of first-line treatment in CRC follows"
results = db.similarity_search(query)
print(results[0].page_content)
```

---

## 🛠 Environment Setup

```bash
pip install -U langchain chromadb faiss-cpu beautifulsoup4 sentence-transformers python-dotenv
```

Also ensure:

* You have the `.env` file with your API keys (if needed).
* You have the `Colon.pdf` and `speech.txt` files in your working directory.

---

## 📂 File Structure
.
├── speech.txt                # Sample text file
├── Colon.pdf                 # Sample PDF document
├── .env                      # API keys and configuration
├── simpleRAG.ipynb            # Main LangChain pipeline script
└── README.md

## 📌 Key Concepts Learned

* 📂 Multiple data loading techniques: local files, web scraping, PDFs
* ✂️ Efficient text chunking for RAG
* 🧬 Embedding text with sentence-transformers
* 📚 Building vector stores with Chroma & FAISS
* 🔍 Semantic search with natural language queries

---

## 📚 Next Steps

* Add memory or context windows for QA systems
* Use `RetrievalQA` with LLMs for true RAG workflows
* Serve as API with `LangServe` or `FastAPI`
* Integrate with a chatbot UI for conversational RAG

---

## 🙌 Credits

* [LangChain Docs](https://docs.langchain.com/)
* [Sentence Transformers](https://www.sbert.net/)
* [ChromaDB](https://www.trychroma.com/)
* [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)

---

## 🧠 Let’s Build Smarter Retrieval Systems Together!
