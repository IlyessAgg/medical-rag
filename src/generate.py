import os
import re
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


def rephrase_query(query, n=3):
    """
    Generate n alternative phrasings of the query using the LLM.
    Returns a list of strings including the original query.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    system_message = (
        "You are a biomedical query rewriting assistant specialized in information retrieval. "
        "Generate diverse reformulations of a biomedical question to improve retrieval performance. "
        "Use varied wording, synonyms, and biomedical terminology when appropriate. "
        "Return ONLY a numbered list of questions."
    )
    user_message = (
        f"Generate {n} different reformulations of the following question:\n"
        f"{query}\n\n"
        f"Requirements:\n"
        f"- Preserve the original meaning\n"
        f"- Use different wording and phrasing\n"
        f"- Include at least one version using more technical or biomedical terminology\n"
        f"- Vary the structure (e.g., full question vs concise form)\n\n"
        f"Return exactly {n} questions, one per line, numbered."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()

    lines = text.split("\n")
    questions = []

    for line in lines:
        # Remove numbering like "1. ", "2) ", "- ", etc.
        cleaned = re.sub(r"^\s*(\d+[\.\)]\s*|\-\s*)", "", line).strip()
        if cleaned:
            questions.append(cleaned)

    # Fallback if model misbehaves
    if len(questions) < n:
        # fallback: split by sentences if needed
        extra = re.split(r"\?\s+", text)
        for q in extra:
            q = q.strip()
            if q and q not in questions:
                questions.append(q if q.endswith("?") else q + "?")

    # Enforce exact n
    questions = questions[:n]

    # Deduplicate
    questions = list(dict.fromkeys(questions))

    return [query] + questions