from src.retrieve import get_collection, retrieve
from src.generate import generate
from src.ingest import get_embedding_model


if __name__ == "__main__":

    collection = get_collection()
    embedding_model = get_embedding_model()

    query = "Do preoperative statins affect outcomes after cardiac surgery?"
    documents = retrieve(query, collection, embedding_model)

    answer = generate(query, documents)

    print(answer)