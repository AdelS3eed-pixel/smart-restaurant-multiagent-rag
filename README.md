# smart-restaurant-multiagent-rag

# Elsada Elafadel — Smart Restaurant Assistant

A multi-agent RAG system for a restaurant chain with four branches (Tahrir, October, Shebin El Kom, Nasr City). The assistant answers questions from an internal knowledge base, checks table availability, handles bookings, and remembers context across a conversation — all routed through a single orchestrator built with **LangGraph**.

**Live demo:** _[add your Streamlit Cloud link here after deployment]_

---

## Architecture

```
User message
     │
     ▼
Orchestrator (LangGraph StateGraph)
     │
     ├─ classify_intent ──► rag ─────────────► RAG Agent ─────► ChromaDB
     │
     ├─ classify_intent ──► operations ──► extract_booking_details ──► Operations Agent ──► Booking Tools
     │
     └─ classify_intent ──► out_of_scope ──► polite decline
```

The orchestrator contains **no business logic itself** — it only classifies intent and routes to the correct sub-agent. All actual knowledge retrieval and tool execution lives inside the sub-agents.

### Why LangGraph

The orchestration logic here (classify → route → respond) is simple enough to write as plain Python `if/else`. This project implements it as a **LangGraph `StateGraph`** instead, using:
- A typed `GraphState` shared across every node
- **Nodes** (`classify_intent`, `extract_booking_details`, `run_rag`, `run_operations`, `run_out_of_scope`) — each a plain Python function
- A **conditional edge** that routes based on the classified intent

This gives the same architecture an explicit, inspectable structure, and makes it straightforward to extend with new nodes (e.g. a cancellation flow) without restructuring the control flow.

### Sub-agents

- **RAG Agent** (`agents/rag_agent.py`) — answers questions from the menu, policies, and restaurant story documents.
- **Operations Agent** (`agents/operations_agent.py`) — calls simulated booking tools and phrases the result naturally.

---

## RAG design decisions

**Knowledge domains covered:** Menu (with allergen and dietary info) + Branch policies / opening hours / events / refunds / loyalty + the restaurant's story ("about us"). This goes beyond the minimum two domains required, since the extra content (about-us, loyalty, delivery) cost little to add and gave the retrieval more realistic ground to stand on.

**Chunking strategy:** Documents are split by their natural record boundaries — one dish entry per chunk in the menu, one section per chunk in policies/about-us — instead of a fixed character length. This keeps a dish's name, price, and branch availability together in the same chunk, which matters directly for accuracy: a fixed-length split could easily cut a price away from its dish name.

**Embedding model:** `all-MiniLM-L6-v2` (via `sentence-transformers`), chosen because it's small (~80MB), runs fully locally with no API cost or extra point of failure, and performs well on short-text semantic similarity — a good fit for dish descriptions and short policy paragraphs.

**Vector database:** ChromaDB, running locally with persistent storage in `chroma_db/`.

**Retrieval strategy:** Top-k semantic search (`k=10`). Branch names are embedded directly in each chunk's text (e.g. "Available in branches: Nasr City only"), so the LLM reasons over branch relevance from the retrieved text itself rather than a strict metadata filter — this was a deliberate simplification given the project's time constraints (see Known Limitations).

**Hallucination prevention:**
- A strict system prompt instructs the RAG agent to answer *only* from retrieved context and to explicitly say when information isn't available, rather than guessing.
- Low temperature (0.3) on the RAG agent's generation call.
- The prompt explicitly instructs the model to flag when a "full list" answer might be incomplete, rather than presenting partial results as complete.

---

## Tool simulation

Two operations tools are implemented in `tools/booking_tools.py`, simulating what a real reservations backend (or MCP server) would expose:

- `check_table_availability(date, time, branch)`
- `book_table(name, date, time, branch, guests)`

These don't call any external service — they simulate realistic responses (including branch validation) as allowed by the assessment ("you may implement functions that simulate server logic").

**Safety validation is handled in code, not left to the LLM alone**, at three points before a booking is confirmed:
1. The branch name is checked against a fixed list of real branches — an unrecognized branch (e.g. a typo, or a branch that doesn't exist) is rejected with a clarification request instead of silently booking elsewhere.
2. All required fields (branch, date, time) must be present before any tool is called.
3. The guest name is sanity-checked via a dedicated LLM call to confirm it's a plausible human name, rejecting joke/garbage input (e.g. "book it for dog") before it reaches the booking tool.

This reflects a deliberate design principle: LLM-based extraction is useful for flexibility, but decisions with real consequences (booking at the wrong branch, booking under a nonsense name) are validated explicitly in code.

---

## Memory design

`memory/conversation_memory.py` implements simple session-based (short-term) memory: a list of `{role, content}` messages held in `st.session_state`, so it persists for the duration of a browser session and resets on refresh. The full history (or a recent window, capped at 10 messages) is passed into both the intent classifier and the booking-detail extractor, so the assistant can follow a booking across multiple turns (e.g. branch mentioned in one message, time in the next).

This is intentionally *not* persistent, long-term memory across sessions — that was out of scope for this assessment.

---

## Example queries and behavior

| Query | Behavior |
|---|---|
| "Do you have vegetarian pasta?" | Answers from menu.txt, distinguishing branches correctly |
| "Is the chicken grilled or fried?" | Distinguishes dishes by name (grilled vs. breaded/fried) |
| "Do you host birthday events at Tahrir?" | Answers from policies.txt; correctly says Nasr City does *not* host events |
| "What's your most popular dish?" | Answers from about_us.txt (Macaroni Bechamel) without inventing sales data |
| "Is there a table available in Shebin El Kom tomorrow at 6pm?" | Routes to Operations Agent, calls `check_table_availability` |
| "Book it for Adel" | Routes to Operations Agent, validates the name, calls `book_table` |
| "Book it for dog" | Rejected — fails the plausible-name check, asks for a real name |
| "What's the weather today?" | Classified as out-of-scope, politely declined |

---

## Known limitations

Documented honestly rather than hidden, given the project's tight timeframe:

1. **Partial results on broad queries.** Questions like "give me the full Nasr City menu" or "what can I eat if I'm allergic to dairy" retrieve only the top-k semantically closest chunks (k=10), not an exhaustive filter over the whole menu. The RAG agent is instructed to flag when a list might be incomplete, but a fully reliable answer to this class of question would need a **structured data tool** (e.g. a `menu_data.json` + a `filter_menu()` tool) rather than semantic search alone. This was a conscious scope cut for time.
2. **No numeric/aggregate tool.** Questions like "what's your most expensive dish" aren't reliably answerable by RAG (semantic search doesn't do numeric comparison). No tool was built for this in the current scope.
3. **Single intent per message.** A message combining two intents (e.g. "what are your hours *and* can you book me a table") is classified as one intent only; the other part is silently dropped. A production version would split such messages or run multiple nodes per turn.
4. **No booking modification/cancellation.** "Change the time to 9pm" creates a new booking rather than modifying the existing one, since no booking store or `cancel_booking` tool was implemented in this scope.
5. **Session-only memory.** Conversation history and bookings are not persisted between sessions or across page refreshes.

---

## Assumptions made

- The restaurant chain ("Elsada Elafadel") and its branches, menu, story, and policies are fictional content created for this assessment.
- Branch names are treated as a fixed, closed list (`Tahrir`, `October`, `Shebin El Kom`, `Nasr City`); any other branch name is explicitly rejected rather than guessed.
- "Booking" always defaults to 2 guests unless the user specifies otherwise.
- LLM: Groq API (`llama-3.3-70b-versatile`) was used for all reasoning steps (classification, extraction, response generation), chosen for its free tier and low latency suited to a live conversational demo.

---

## Project structure

```
├── app.py                      # Streamlit UI
├── orchestrator.py             # LangGraph StateGraph — intent routing
├── agents/
│   ├── rag_agent.py            # Knowledge base Q&A
│   └── operations_agent.py     # Booking tool execution + phrasing
├── rag/
│   ├── ingest.py                # One-time document chunking + embedding
│   └── retriever.py             # ChromaDB query interface
├── tools/
│   └── booking_tools.py         # Simulated availability + booking tools
├── memory/
│   └── conversation_memory.py   # Session-based chat history
├── data/
│   ├── menu.txt
│   ├── policies.txt
│   └── about_us.txt
└── chroma_db/                   # Generated by ingest.py (not committed)
```

---

## Running locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Add your Groq API key to a .env file:
# GROQ_API_KEY=your_key_here

python rag/ingest.py           # one-time: build the vector database
streamlit run app.py
```
