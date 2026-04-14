from fastapi import FastAPI
from pydantic import BaseModel
from src.ingest import get_embedding_model
from src.retrieve import get_collection, retrieve
from src.generate import generate

app = FastAPI(title="Medical RAG")

# Load at startup, not per request
embedding_model = get_embedding_model()
collection = get_collection()


class QueryRequest(BaseModel):
    question: str
    n_results: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    
    query = request.question
    n_docs = request.n_results
    
    documents = retrieve(query, collection, embedding_model, n_docs)
    answer = generate(query, documents)

    sources = [
        {key: doc[key] for key in {'pubid', 'question'} if key in doc}
        for doc in documents
    ]

    return QueryResponse(answer=answer, sources=sources)