class ConversationMemory:
    """
    Stores the conversation history for the current session.
    This is short-term (session) memory: it lives only while the
    Streamlit app is running and resets when the session ends.
    """

    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        """role is either 'user' or 'assistant'."""
        self.messages.append({"role": role, "content": content})

    def get_history(self):
        """Returns the full conversation history as a list of dicts."""
        return self.messages

    def get_recent_history(self, max_messages=10):
        """
        Returns only the last N messages, to avoid sending an
        unbounded amount of text to the LLM in long conversations.
        """
        return self.messages[-max_messages:]

    def clear(self):
        self.messages = []