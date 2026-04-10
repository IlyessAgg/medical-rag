# Medical RAG

> Building a Retrieval-Augmented Generation (RAG) system on [PubMed](https://pubmedqa.github.io/index.html) data.

Building on what I learned with the [medical-abstract-classifier](https://github.com/IlyessAgg/medical-abstract-classifier/tree/main), the goal of this project is to explore **Retrieval-Augmented Generation (RAG)**: embedding medical answers, storing them in a vector database *(ChromaDB)*, retrieving relevant context, and generating responses using the *Groq* API.  

The main new things here are working with **vector stores** and retrieval-focused embeddings, while reusing tools I’m already familiar with like FastAPI, Docker, and CI.