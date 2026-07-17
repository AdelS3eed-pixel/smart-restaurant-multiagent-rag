"""
Orchestrator: classifies user intent and routes to the correct sub-agent,
implemented as a LangGraph StateGraph.

LangGraph concepts used here:
- State: a shared dict that flows through every node. Each node reads
  from it and writes back into it.
- Node: a Python function that does one job (classify, call an agent,
  extract booking details...) and returns updates to the state.
- Edge: a connection between nodes. A "conditional edge" picks the next
  node based on a value in the state (here: the classified intent).
- The graph is built once (nodes + edges), then "compiled" into a
  runnable object we call for every user message.

This graph mirrors exactly the same logic that would otherwise be written
as plain if/else Python: classify -> route -> respond. LangGraph just
gives that flow an explicit, inspectable structure.
"""

import os
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from groq import Groq

from agents.rag_agent import RAGAgent
from agents.operations_agent import OperationsAgent

VALID_BRANCHES = ["Tahrir", "October", "Shebin El Kom", "Nasr City"]


# ---- State definition ----
# This is the shared "memory" that flows through the graph for a single turn.
class GraphState(TypedDict):
    question: str
    history: list
    intent: Optional[str]
    branch: Optional[str]
    date: Optional[str]
    time: Optional[str]
    guests: Optional[int]
    name: Optional[str]
    wants_to_book: Optional[bool]
    answer: Optional[str]


class Orchestrator:
    def __init__(self, groq_api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.rag_agent = RAGAgent(self.client, model)
        self.operations_agent = OperationsAgent(self.client, model)
        self.graph = self._build_graph()

    # ---- Nodes ----

    def _classify_intent(self, state: GraphState) -> GraphState:
        """
        Node 1: classify the user's question into one of three intents.
        This is the only place an LLM call decides "what kind of question is this".
        """
        history = state.get("history", [])

        # Deterministic override: if the assistant's last message was asking
        # for booking details (branch/date/time/name), the next user message
        # is almost certainly continuing that booking, even if it's a short
        # or unusual reply (like a name). Don't let the LLM misclassify it
        # as out_of_scope just because the content looks odd on its own.
        if history:
            last_message = history[-1]
            if last_message.get("role") == "assistant":
                last_text = last_message.get("content", "").lower()
                booking_prompts = ["which branch", "the date", "the time", "what name"]
                if any(phrase in last_text for phrase in booking_prompts):
                    state["intent"] = "operations"
                    return state

        prompt = f"""Classify the user's message into exactly one category:
- "rag": a question about the menu, dishes, ingredients, allergens, opening
  hours, policies, refunds, events, loyalty program, or the restaurant's story.
- "operations": a request to check table availability or make a booking.
- "out_of_scope": anything unrelated to the restaurant (weather, politics,
  general knowledge, etc.)

If the assistant's previous message was asking the user for booking details
(branch, date, time, or name), and the user's message could plausibly be
answering that question, classify it as "operations" even if the content
looks unusual on its own.

Conversation so far:
{self._format_history(history)}

User message: "{state['question']}"

Respond with ONLY one word: rag, operations, or out_of_scope."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        intent = response.choices[0].message.content.strip().lower()

        if intent not in ("rag", "operations", "out_of_scope"):
            intent = "rag"  # safe fallback

        state["intent"] = intent
        return state

    def _extract_booking_details(self, state: GraphState) -> GraphState:
        """
        Node 2 (operations branch only): pull date/time/branch/guests/name
        out of the user's message (and recent history) using the LLM.
        """
        prompt = f"""Extract booking details by combining information from the user's
messages below (ignore anything the assistant said - only trust what the
USER explicitly typed).

Return ONLY a JSON object with these exact keys: branch, date, time, guests, name, wants_to_book.
- Use null for any value the user never explicitly stated themselves.
- For "branch": use the MOST RECENTLY mentioned branch name that the USER
  (not the assistant) wrote, exactly as they wrote it, even if it's not a
  known branch name. Never infer a branch from the assistant's replies.
- If the user's latest message starts a new booking request (e.g. mentions
  a different branch, date, or says "book a table" again) after a previous
  booking was already confirmed, treat this as a FRESH request: ignore
  date/time/branch from before that confirmed booking.
- Normalize relative dates (e.g. "24th", "tomorrow") into a clear date string as written by the user.
- wants_to_book must be true if the user explicitly wants to RESERVE/BOOK a table
  (e.g. "book", "reserve", "I want a table"), or false if they are only asking
  whether tables are available (e.g. "is there availability", "do you have a table").

Full conversation (oldest to newest):
{self._format_history(state.get("history", []))}

Latest message: "{state['question']}"

Respond with ONLY the JSON object, no explanation, no markdown code fences.

JSON:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        import json
        try:
            details = json.loads(response.choices[0].message.content.strip())
        except Exception:
            details = {}

        state["branch"] = details.get("branch")
        state["date"] = details.get("date")
        state["time"] = details.get("time")
        state["guests"] = details.get("guests") or 2
        state["name"] = details.get("name")
        state["wants_to_book"] = bool(details.get("wants_to_book"))
        return state

    def _run_rag(self, state: GraphState) -> GraphState:
        """Node 3a: hand off to the RAG agent."""
        state["answer"] = self.rag_agent.answer(state["question"])
        return state

    def _run_operations(self, state: GraphState) -> GraphState:
        """
        Node 3b: hand off to the operations agent, once we have enough
        details. If something required is missing, ask for it instead
        of calling the tool with incomplete data.
        """
        missing = []
        branch = state.get("branch")

        # Explicit code-level validation - never trust the LLM alone for
        # something this important. If the branch name isn't one of our
        # real branches, stop here and ask for clarification instead of
        # silently booking at a different (possibly stale) branch.
        if branch and branch not in VALID_BRANCHES:
            state["answer"] = (
                f"We don't have a branch called '{branch}'. "
                f"Our branches are: {', '.join(VALID_BRANCHES)}. Which one would you like?"
            )
            return state

        if not branch:
            missing.append("which branch")
        if not state.get("date"):
            missing.append("the date")
        if not state.get("time"):
            missing.append("the time")

        if missing:
            state["answer"] = (
                "Happy to help with that! Could you tell me " + ", ".join(missing) + "?"
            )
            return state

        # If the user clearly wants to book but hasn't given a name yet,
        # ask for the name specifically instead of silently treating this
        # as a plain availability check.
        if state.get("wants_to_book") and not state.get("name"):
            state["answer"] = "Great, I can book that for you! What name should I put the reservation under?"
            return state

        # Validate that the "name" actually looks like a plausible human
        # name before using it. This is a booking under a real guest's
        # name, not free-form text - garbage or joke input shouldn't be
        # accepted silently.
        if state.get("wants_to_book") and state.get("name"):
            if not self._looks_like_a_name(state["name"]):
                state["answer"] = (
                    f"'{state['name']}' doesn't look like a valid guest name. "
                    "Could you provide the name the reservation should be under?"
                )
                return state

        if state.get("wants_to_book") and state.get("name"):
            state["answer"] = self.operations_agent.make_booking(
                name=state["name"],
                date=state["date"],
                time=state["time"],
                branch=state["branch"],
                guests=state["guests"]
            )
        else:
            state["answer"] = self.operations_agent.check_availability(
                date=state["date"],
                time=state["time"],
                branch=state["branch"]
            )

        return state

    def _looks_like_a_name(self, name: str) -> bool:
        """
        Quick sanity check that the extracted 'name' is a plausible human
        name, not a joke, an animal, gibberish, or unrelated text. Uses a
        cheap, deterministic-style LLM call (temperature=0) since this
        judgment doesn't have a reliable rule-based check.
        """
        prompt = f"""Is "{name}" a plausible human first name or full name that
a restaurant could reasonably put on a table reservation?
Answer with ONLY "yes" or "no". If it's an animal, an object, a joke,
gibberish, or clearly not a person's name, answer "no"."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower().startswith("yes")

    def _run_out_of_scope(self, state: GraphState) -> GraphState:
        """Node 3c: politely decline unrelated questions."""
        state["answer"] = (
            "I'm the Elsada Elafadel restaurant assistant, so I can only help "
            "with questions about our menu, policies, or table bookings. "
            "Is there something along those lines I can help with?"
        )
        return state

    # ---- Routing ----

    def _route_by_intent(self, state: GraphState) -> str:
        """Conditional edge: picks the next node based on the classified intent."""
        return state["intent"]

    # ---- Graph construction ----

    def _build_graph(self):
        graph = StateGraph(GraphState)

        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("extract_booking_details", self._extract_booking_details)
        graph.add_node("run_rag", self._run_rag)
        graph.add_node("run_operations", self._run_operations)
        graph.add_node("run_out_of_scope", self._run_out_of_scope)

        graph.set_entry_point("classify_intent")

        graph.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "rag": "run_rag",
                "operations": "extract_booking_details",
                "out_of_scope": "run_out_of_scope",
            }
        )

        graph.add_edge("extract_booking_details", "run_operations")
        graph.add_edge("run_rag", END)
        graph.add_edge("run_operations", END)
        graph.add_edge("run_out_of_scope", END)

        return graph.compile()

    # ---- Public entry point ----

    def handle_message(self, question: str, history: list) -> str:
        initial_state: GraphState = {
            "question": question,
            "history": history,
            "intent": None,
            "branch": None,
            "date": None,
            "time": None,
            "guests": None,
            "name": None,
            "wants_to_book": None,
            "answer": None,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state["answer"]

    @staticmethod
    def _format_history(history: list) -> str:
        if not history:
            return "(no previous messages)"
        lines = [f"{m['role']}: {m['content']}" for m in history[-6:]]
        return "\n".join(lines)