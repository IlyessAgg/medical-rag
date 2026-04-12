import chromadb
from sentence_transformers import SentenceTransformer
from src.ingest import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from pprint import pprint


def get_collection():
    """Load the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def retrieve(query, collection, embedding_model, n_results=3):
    """
    Embed the query and retrieve the n most similar documents.
    
    Returns a list of dicts, each with:
    - 'text': the document text
    - 'question': the original question from metadata
    - 'pubid': the pubid from metadata
    - 'distance': the similarity distance
    """
    query_embedding = embedding_model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    retrieved_docs = []
    for i in range(len(results["documents"][0])):
        retrieved_docs.append({
            "text": results["documents"][0][i],
            "question": results["metadatas"][0][i]["question"],
            "pubid": results["metadatas"][0][i]["pubid"],
            "distance": results["distances"][0][i],
        })

    return retrieved_docs


if __name__ == "__main__":
    print("Loading collection...")
    collection = get_collection()
    print("Loaded collection.")

    query = "Do statins reduce cardiovascular risk?"
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    documents = retrieve(query, collection, embedding_model)

    print(f"\nQuery: {query}\n")
    pprint(documents)
