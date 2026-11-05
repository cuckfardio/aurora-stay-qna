
import os
from types import SimpleNamespace

import anthropic


class ClaudeChat:
    """Minimal chat client: .invoke(prompt).content, backed by Claude through the Anthropic SDK."""

    THINKING_HEADROOM = 2048  # Opus 5.5 always reasons first; reserve room so the answer is not cut off

    def __init__(self, model, max_tokens=512, effort="low"):
        self.model, self.max_tokens, self.effort = model, max_tokens, effort
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def invoke(self, prompt):
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens + self.THINKING_HEADROOM,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return SimpleNamespace(content="The request was declined by the model's safeguards; please look this up manually.")
        return SimpleNamespace(content="".join(b.text for b in response.content if b.type == "text").strip())

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant helping AuroraStay Hotels staff answer guest questions using AuroraStay's internal hotel policy documents.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exclusions, limitations, exceptions, or coverage details.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent fees, amounts, times, deadlines, property names, exceptions, or approval steps.
    11. When a rule differs by property or rate plan, state the value for the property or rate plan asked about.
    12. Answer in plain prose a staff member can read to a guest; do not mention chunk numbers or metadata.
    10. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "claude-opus-5-5",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    # temperature / top_p stay in the API contract but are not sent: Opus 5.5 rejects sampling parameters
    model = ClaudeChat(model=model_name, max_tokens=max_tokens)

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
