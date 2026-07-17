"""
RAG Agent: answers questions using retrieved context from the knowledge base.
Uses a strict system prompt to prevent hallucination - it must only use
the retrieved context, and must say so clearly if the answer isn't there.
"""

from groq import Groq
from rag.retriever import retrieve_as_context

SYSTEM_PROMPT = """You are the knowledge assistant for Elsada Elafadel restaurant chain.
Answer the user's question using ONLY the context provided below. The context
comes from our menu, policies, and about-us documents.

Rules:
- Only use information explicitly present in the context.
- If the answer is not in the context, say clearly that you don't have that
  information, instead of guessing or inventing details.
- If a dish or policy is only available at certain branches, mention that
  explicitly in your answer.
- If the user asks for a full or complete menu/list and the context you were
  given might only be a partial selection, clearly say that this may not be
  the full list, instead of presenting it as complete.
- Keep answers concise, friendly, and in English.

Context:
{context}
"""


class RAGAgent:
    def __init__(self, groq_client: Groq, model: str = "llama-3.3-70b-versatile"):
        self.client = groq_client
        self.model = model

    def answer(self, question: str) -> str:
        context = retrieve_as_context(question, n_results=10)
        system_message = SYSTEM_PROMPT.format(context=context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content