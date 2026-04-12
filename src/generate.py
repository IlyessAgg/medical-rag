import os
from groq import Groq

MODEL = "llama-3.3-70b-versatile"


def build_prompt(query, retrieved_docs):
    """
    Build a prompt that includes the query and retrieved context.
    
    Format:
    - A system message defining the assistant's role
    - A user message containing:
        - The retrieved documents as numbered context passages
        - The question to answer
    
    Returns a tuple: (system_message, user_message) both as strings.
    """

    system_message = (
        "You are a medical assistant. Answer questions using ONLY the provided context. "
        "If the answer is not contained in the context, say you don't know. "
        "Be concise and accurate."
    )

    context = ""
    for i, doc in enumerate(retrieved_docs, 1):
        context += f"Document {i}:\n{doc['text']}\n\n"

    user_message = (
        f"Context:\n{context}\n"
        f"Question: {query}\n"
        "Answer:"
    )

    return system_message, user_message


def generate(query, retrieved_docs):
    """
    Call the Groq API with the built prompt.
    
    Returns the generated answer as a string.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    system_message, user_message = build_prompt(query, retrieved_docs)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content