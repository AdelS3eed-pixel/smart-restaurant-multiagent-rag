"""
Operations Agent: handles booking-related requests by calling the
simulated booking tools, then phrasing the tool's result as a natural
language response.
"""

from groq import Groq
from tools.booking_tools import check_table_availability, book_table

SYSTEM_PROMPT = """You are the operations assistant for Elsada Elafadel restaurant chain.
You just received the result of a tool call (either a table availability
check or a booking confirmation). Turn the raw result below into a short,
friendly, natural language response for the customer, in English.

Do not invent any information beyond what's in the tool result.

Tool result:
{tool_result}
"""


class OperationsAgent:
    def __init__(self, groq_client: Groq, model: str = "llama-3.3-70b-versatile"):
        self.client = groq_client
        self.model = model

    def check_availability(self, date: str, time: str, branch: str) -> str:
        result = check_table_availability(date, time, branch)
        return self._phrase_result(result)

    def make_booking(self, name: str, date: str, time: str, branch: str, guests: int = 2) -> str:
        result = book_table(name, date, time, branch, guests)
        return self._phrase_result(result)

    def _phrase_result(self, tool_result: dict) -> str:
        system_message = SYSTEM_PROMPT.format(tool_result=tool_result)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Please phrase this result for the customer."}
            ],
            temperature=0.4
        )

        return response.choices[0].message.content