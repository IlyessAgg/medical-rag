# Medical RAG

> Building a Retrieval-Augmented Generation (RAG) system on [PubMed](https://pubmedqa.github.io/index.html) data.

Building on what I learned with the [medical-abstract-classifier](https://github.com/IlyessAgg/medical-abstract-classifier/tree/main), this project explores **Retrieval-Augmented Generation (RAG)**: embedding medical documents, storing them in a vector database (*ChromaDB*), retrieving relevant context, and generating grounded answers using the *Groq* API.

The main new things here are working with **vector stores** and retrieval-focused embeddings, while reusing tools I’m already familiar with like FastAPI, Docker, and CI.

## Dataset & Preprocessing

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

## Embeddings

We use : `pritamdeka/S-PubMedBert-MS-MARCO`

This model is specifically tuned for **biomedical semantic search**, making it well-suited for retrieval tasks.

Documents are embedded in batches and stored in a vector database.

## Vector Store (ChromaDB)

We use **ChromaDB** as a persistent vector database:

- Collection name: `pubmedqa`
- Storage path: `data/chroma`
- Embeddings: dense vectors from `SentenceTransformers`

> [!NOTE] Distance Metric
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

## Generation (Groq API)

We use the Groq API for fast inference with a **low temperature (0.2)**:

- Ensures deterministic, factual responses.
- Reduces *hallucinations*.
- Better suited for **medical QA**.


## Grounded Answering

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


## Project Structure

```
├── data/  
│   └── chroma/           # Persistent ChromaDB storage  
├── src/  
│   ├── ingest.py         # Load dataset + build vector store  
│   ├── retrieve.py       # Query embedding + similarity search  
│   └── generate.py       # LLM generation with Groq  
├── .env                  # API keys (GROQ_API_KEY)  
├── main.py               # Entry point / pipeline orchestration  
├── requirements.txt  
└── README.md
```

## Environment Variables

Create a `.env` file :

```bash
GROQ_API_KEY=your_api_key_here
```

Used in `src/generate.py` as :

```python
import os  
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
```