import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
COLLECTION_NAME = "pubmedqa"
CHROMA_PATH = "data/chroma"


def load_documents():
    """
    Load PubMedQA and extract documents.
    
    Returns a list of dicts, each with:
    - 'id': str(pubid)
    - 'text': the long_answer
    - 'question': the original question
    """
    dataset = load_dataset("pubmed_qa", "pqa_labeled")

    docs = [
        {
            "id": str(example["pubid"]),
            "text": example["long_answer"],
            "question": example["question"],
        }
        for example in dataset['train']
    ]

    return docs


def get_embedding_model():
    """Load and return the sentence transformer model."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def build_vector_store(documents, embedding_model):
    """
    Embed documents and store in ChromaDB.
    
    - Create a persistent ChromaDB client at CHROMA_PATH
    - Create or get collection named COLLECTION_NAME
    - Embed each document's text
    - Add to collection with ids, embeddings, documents, and metadatas
    
    Metadata per document: {'question': ..., 'pubid': ...}
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        print("Collection already exists, skipping ingestion.")
        return

    texts = [doc["text"] for doc in documents]
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True)

    ids = [doc["id"] for doc in documents]
    documents_text = [doc["text"] for doc in documents]
    metadatas = [
        {"question": doc["question"], "pubid": doc["id"]}
        for doc in documents
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents_text,
        metadatas=metadatas,
    )


if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("Loading embedding model...")
    model = get_embedding_model()

    print("Building vector store...")
    build_vector_store(documents, model)
    print(f"Vector store saved to {CHROMA_PATH}")