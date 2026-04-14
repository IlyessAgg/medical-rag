# Medical RAG

> Building a Retrieval-Augmented Generation (RAG) system on [PubMed](https://pubmedqa.github.io/index.html) data.

<div align="center">

<p align="center">
  <img src="assets/soul-terry.gif" alt="Searching..." width="400"/>
</p>

![CI](https://github.com/IlyessAgg/medical-abstract-classifier/workflows/CI/badge.svg)

</div>

Building on what I learned with the [medical-abstract-classifier](https://github.com/IlyessAgg/medical-abstract-classifier/tree/main), this project explores **Retrieval-Augmented Generation (RAG)**: embedding medical documents, storing them in a vector database (*ChromaDB*), retrieving relevant context, and generating grounded answers using the *Groq* API.

A key addition to this project is improving retrieval quality through **multi-query retrieval**, where the original question is reformulated into multiple semantic variations to improve recall in the vector space.

The main new things here are working with **vector stores**, **retrieval strategies (single-query vs multi-query)**, and embedding-based search, while reusing tools I’m already familiar with like FastAPI, Docker, and CI.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Getting Started](#getting-started)
  - [1. Install dependencies](#1-install-dependencies)
  - [2. Set environment variables](#2-set-environment-variables)
  - [3. Build the vector database](#3-build-the-vector-database)
  - [4. Run the API](#4-run-the-api)
  - [5. Or run with Docker](#5-or-run-with-docker)
- [Querying the API](#querying-the-api)
  - [Example response](#example-response)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Dataset & Preprocessing](#dataset--preprocessing)
  - [Embeddings](#embeddings)
  - [Vector Store (ChromaDB)](#vector-store-chromadb)
  - [Generation (Groq API)](#generation-groq-api)
  - [Grounded Answering](#grounded-answering)
- [Multi-Query Retrieval](#multi-query-retrieval)
  - [Motivation](#motivation)
  - [Implementation](#implementation)
  - [Observed Behavior](#observed-behavior)
    - [1. Weak or underspecified queries](#1-weak-or-underspecified-queries)
    - [2. Semantically biased queries](#2-semantically-biased-queries)
    - [3. Well-aligned queries](#3-well-aligned-queries)
  - [Multi-Query vs Increasing k](#multi-query-vs-increasing-k)
  - [Tradeoffs](#tradeoffs)
  - [Key Takeaway](#key-takeaway)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Getting Started

The vector database in `data/chroma/` is **not versioned**. 
You must rebuild it locally before running the API or Docker container.

Follow these steps in order:  

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file: 

```
GROQ_API_KEY=your_api_key_here
```

### 3. Build the vector database

Run the ingestion script to populate ChromaDB:

```bash
python ingest.py
```

This will create the `data/chroma/` directory locally.

### 4. Run the API

```bash
uvicorn src.main:app --reload
```

### 5. Or run with Docker

```bash
docker build -t medical-rag .
docker run --env-file .env -p 8000:8000 medical-rag
```

## Querying the API

You can interact with the model using the `/query` endpoint:

```shell
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "Do preoperative statins affect outcomes after cardiac surgery?",
  "n_results": 3,
  "technique": "standard"
}'
```

The possible arguments for `technique` are `standard` or `multi_query`.

### Example response

```json
{
  "answer": "According to Document 1, preoperative statin therapy seems to reduce the development of atrial fibrillation (AF) after coronary artery bypass grafting (CABG).",
  "sources": [
    {
      "question": "...",
      "pubid": 12345
    },
    ...
  ]
}
```

## Project Structure

```
├── data/  
│   └── chroma/           # Persistent ChromaDB storage  
├── src/  
│   ├── api.py            # FastAPI app and endpoint definitions
│   ├── ingest.py         # Load dataset + build vector store  
│   ├── retrieve.py       # Query embedding + similarity search  
│   └── generate.py       # LLM generation with Groq  
├── .github/              
│   └── workflows/        # CI/CD workflows (e.g., ci.yml)  
├── tests/                
│   └── test_data.py      # Unit tests for pipeline
├── .env                  # API keys (GROQ_API_KEY)  
├── requirements.txt  
├── Dockerfile            # Docker configuration for containerizing the app
├── main.py               # Entry point / pipeline orchestration
├── assets/               # Images and other assets for the README
└── README.md
```

## How It Works

### Dataset & Preprocessing

This project uses the [labeled PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) dataset, but instead of framing it as a classification task, we treat it as a **document retrieval problem**.

Each example is transformed into a document with: 
- `id`: PubMed ID
- `text`: the **long answer** (used as retrieval context)
- `question`: the original query (stored as metadata)

This allows us to build a **semantic search system over expert-written medical answers**.

Key differences from the classifier project:
- We **do not use class labels** (`yes/no/maybe`).
- We embed *long-form answers* instead of *abstracts*.
- We optimize for **retrieval similarity**, not classification accuracy.

### Embeddings

We use : `pritamdeka/S-PubMedBert-MS-MARCO`

This model is specifically tuned for **biomedical semantic search**, making it well-suited for retrieval tasks.

Documents are embedded in batches and stored in a vector database.

### Vector Store (ChromaDB)

We use **ChromaDB** as a persistent vector database:

- Collection name: `pubmedqa`
- Storage path: `data/chroma`
- Embeddings: dense vectors from `SentenceTransformers`

> [!NOTE] 
> **Distance Metric**  
> By default, ChromaDB uses **L2 distance (Euclidean distance)** *(lower = more similar)*.
>
> This is important when interpreting retrieval results.  
> If cosine similarity were preferred, it could be configured at collection creation:
> ```python
>collection = client.get_or_create_collection(
>    name=COLLECTION_NAME,
>    metadata={"hnsw:space": "cosine"}
>)
>```

### Generation (Groq API)

We use the Groq API for fast inference with a **low temperature (0.2)**:

- Ensures deterministic, factual responses.
- Reduces *hallucinations*.
- Better suited for **medical QA**.


### Grounded Answering

A key design choice in this project is **strict grounding in retrieved context**.

If the answer cannot be found in the retrieved documents, the model responds:

> _"I don't know."_

This is **intentional and desirable**, especially in a medical setting:

- Prevents hallucinated or unsafe answers.
- Reflects real limitations of the dataset.
- Encourages trustworthiness over completeness.

**Example 1**
```text
Query: Do statins reduce cardiovascular risk?

Answer: I don't know. 
```

Why? The retrieved documents discuss **specific contexts** (e.g., stroke, CABG outcomes), not the general question.

**Example 2**

```text
Query: Do preoperative statins affect outcomes after cardiac surgery?

Answer: Yes, preoperative statin therapy seems to reduce AF development after CABG.
```

This works because the query **closely matches the underlying documents**.

## Multi-Query Retrieval  
  
To improve retrieval robustness, we optionally use a **multi-query strategy**.  
  
Instead of embedding only the original query, we first generate several **alternative reformulations** using an LLM, then retrieve documents for each of them.  
  
### Motivation  
  
Dense retrieval models are sensitive to wording. Two semantically equivalent questions can map to different regions of the embedding space.  
  
Multi-query retrieval addresses this by:  
- Expanding the query into multiple **semantic variations**  
- Increasing the chance of matching relevant documents  
- Reducing reliance on a single phrasing  

### Implementation  
  
Given a query, we generate *n* reformulations:  
  
```python  
queries = [original_query] + rephrase_query(original_query, n=3)
```

Each query is then used independently:

```python
for q in queries:
  results = retrieve(q, collection, embedding_model, n_results=n_results)
```

Results are then:  

- **Merged**
- **Deduplicated** (by document content or ID)


### Observed Behavior

We compared standard retrieval (single query) with multi-query retrieval across several biomedical questions.

#### 1. Weak or underspecified queries

Example:

> _“Do statins reduce inflammation?”_

- Single-query retrieval focused on cardiovascular outcomes
- Multi-query retrieval surfaced additional **immunology-related documents**

**Insight:**  
Multi-query improves recall when the original query does not align well with the dataset vocabulary.

#### 2. Semantically biased queries

Example:

> _“Is aspirin effective for preventing heart attacks?”_

- Single-query retrieval focused on **treatment of acute myocardial infarction**
- Increasing `k` did not recover prevention-related documents
- Multi-query retrieval expanded into **broader cardiovascular contexts**

**Insight:**  
Multi-query can explore **different semantic directions**, not just a larger neighborhood.

#### 3. Well-aligned queries

Example:

> _“Does vitamin D improve immune response?”_

- Single-query retrieval already returned a highly relevant document
- Multi-query added marginal results, mostly increasing noise

**Insight:**  
When the query is already well-formed, multi-query provides limited benefit.

### Multi-Query vs Increasing k

We also compared multi-query retrieval with simply increasing the number of retrieved documents (`k`).

- Increasing `k` expands retrieval **locally** (same semantic region)
- Multi-query expands retrieval **directionally** (different semantic regions)

In some cases, increasing `k` can approximate multi-query results.  
However, it fails when the original query is **semantically biased or incomplete**.

### Tradeoffs

**Advantages**

- Higher recall
- Better handling of biomedical terminology variation
- More robust to query phrasing

**Limitations**

- Increased latency (multiple LLM calls)
- More noisy results
- Requires deduplication (and ideally reranking)

### Key Takeaway

Multi-query retrieval is **not universally beneficial**.

It is most useful when:

- the query is ambiguous
- the query uses non-clinical wording
- or the query does not match dataset terminology

When the query is already well-aligned, standard retrieval is often sufficient.
