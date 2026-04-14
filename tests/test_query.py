from src.retrieve import retrieve
from src.generate import build_prompt
from unittest.mock import Mock


def test_build_prompt():
    query = "What is diabetes ?"

    system_message, user_message = build_prompt(
        query,
        [{"text": "Diabetes is a chronic condition."}]
    )

    assert "ONLY" in system_message
    assert query in user_message


def test_retrieve():
    query = "What is diabetes ?"

    embedding_model = Mock()
    embedding_model.encode.return_value = [0.1, 0.2, 0.3]

    collection = Mock()
    collection.query.return_value = {
        "documents": [["doc1 text", "doc2 text"]],
        "metadatas": [[
            {"question": "Q1", "pubid": 1},
            {"question": "Q2", "pubid": 2}
        ]],
        "distances": [[0.1, 0.2]]
    }

    documents = retrieve(query, collection, embedding_model)

    assert len(documents) == 2

    assert documents[0] == {
        "text": "doc1 text",
        "question": "Q1",
        "pubid": 1,
        "distance": 0.1,
    }

    assert documents[1]["text"] == "doc2 text"
    assert documents[1]["question"] == "Q2"