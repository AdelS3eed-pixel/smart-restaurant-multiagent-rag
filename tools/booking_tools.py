"""
Simulated operations tools for table booking.
These functions mimic what a real restaurant booking system (POS/MCP server)
would expose. They don't call any external service - they simulate the
behavior as required by the assessment.
"""

import random

VALID_BRANCHES = ["Tahrir", "October", "Shebin El Kom", "Nasr City"]


def check_table_availability(date: str, time: str, branch: str) -> dict:
    """
    Simulates checking table availability at a given branch, date, and time.
    Returns a dict with availability info.
    """
    if branch not in VALID_BRANCHES:
        return {
            "success": False,
            "message": f"'{branch}' is not a valid branch. Valid branches are: {', '.join(VALID_BRANCHES)}."
        }

    # Simulated logic: randomly decide availability, but keep it deterministic-ish
    tables_left = random.randint(0, 6)
    available = tables_left > 0

    return {
        "success": True,
        "branch": branch,
        "date": date,
        "time": time,
        "available": available,
        "tables_left": tables_left,
        "message": (
            f"Tables are available at {branch} on {date} at {time}."
            if available else
            f"Sorry, no tables are available at {branch} on {date} at {time}."
        )
    }


def book_table(name: str, date: str, time: str, branch: str, guests: int = 2) -> dict:
    """
    Simulates booking a table. In a real system this would write to a
    database or call an external reservations API/MCP server.
    """
    if branch not in VALID_BRANCHES:
        return {
            "success": False,
            "message": f"'{branch}' is not a valid branch. Valid branches are: {', '.join(VALID_BRANCHES)}."
        }

    booking_id = random.randint(1000, 9999)

    return {
        "success": True,
        "booking_id": booking_id,
        "name": name,
        "branch": branch,
        "date": date,
        "time": time,
        "guests": guests,
        "message": (
            f"Booking confirmed for {name} at {branch} on {date} at {time} "
            f"for {guests} guest(s). Booking ID: {booking_id}."
        )
    }