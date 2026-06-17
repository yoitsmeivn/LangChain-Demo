"""
Chinook Music Store — Customer Support Bot
"""

from pathlib import Path
import sqlite3

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.config import get_store, get_config  # store memory on RAM

from typing_extensions import TypedDict
from langchain.agents.middleware import dynamic_prompt


# --- database help ---
DB_PATH = Path(__file__).parent.parent.parent / "chinook-db" / "chinook.db"

def run_sql(sql: str, params: tuple = ()) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- model ---
llm = ChatAnthropic(model="claude-sonnet-4-5")

# --- tools: 4 (catalog + account) ---
@tool
def search_tracks(query: str) -> list:
    """Search for tracks by name, artist, or album."""
    return run_sql(
        """
        SELECT t.Name AS track, ar.Name AS artist, al.Title AS album
        FROM Track t
        JOIN Album al ON t.AlbumId = al.AlbumId
        JOIN Artist ar ON al.ArtistId = ar.ArtistId
        WHERE t.Name LIKE ? OR ar.Name LIKE ?
        LIMIT 8
        """,
        (f"%{query}%", f"%{query}%"),
    )

@tool
def recommend_by_genre(genre: str) -> list:
    """Recommend tracks for a genre like rock, jazz, or pop."""
    return run_sql(
        """
        SELECT t.Name AS track, ar.Name AS artist
        FROM Track t
        JOIN Album al ON t.AlbumId = al.AlbumId
        JOIN Artist ar ON al.ArtistId = ar.ArtistId
        JOIN Genre g ON t.GenreId = g.GenreId
        WHERE g.Name LIKE ?
        ORDER BY RANDOM() LIMIT 6
        """,
        (f"%{genre}%",),
    )

@tool
def get_my_orders() -> list:
    """Get the logged-in customer's purchase history."""
    email = get_config()["configurable"]["customer_email"]
    return run_sql(
        """
        SELECT i.InvoiceId, i.InvoiceDate, i.Total
        FROM Invoice i
        JOIN Customer c ON i.CustomerId = c.CustomerId
        WHERE c.Email = ?
        ORDER BY i.InvoiceDate DESC LIMIT 10
        """,
        (email,),
    )

@tool
def get_my_account() -> dict:
    """Get the logged-in customer's account info."""
    email = get_config()["configurable"]["customer_email"]
    rows = run_sql(
        "SELECT FirstName, LastName, Email, Country FROM Customer WHERE Email = ?",
        (email,),
    )
    return rows[0] if rows else {"error": "not found"}

@tool
def remember_preference(key: str, fact: str) -> str:
    """Save a lasting fact about THIS customer, e.g. key='genre', fact='jazz'."""
    store = get_store()
    store.put(_customer_ns(), key, {"fact": fact})
    return f"Saved {key}: {fact}"

@tool
def recall_preferences() -> str:
    """Load everything saved about THIS customer."""
    store = get_store()
    items = store.search(_customer_ns())
    if not items:
        return "No saved preferences yet."
    return "; ".join(f"{it.key}: {it.value.get('fact', it.value)}" for it in items)


# --- system prompt ---
SYSTEM = """You are a support agent for the Chinook Music Store.

GREETING:
- On your VERY FIRST reply in a conversation, begin with this one-line welcome, then immediately answer whatever the customer asked:
  "Hi! I'm the Chinook Music Store support assistant — happy to help with music or your account."
- Do this only once, on the first reply. Never greet again later in the same conversation.

You help with: 1) MUSIC — search and recommend by genre. 2) ACCOUNT — orders and account details.

MEMORY RULES:
- NEVER state a customer's saved preferences from memory of this conversation. ALWAYS call recall_preferences first and answer ONLY from its result.
- If recall_preferences returns "No saved preferences yet," tell the customer nothing is saved — do not infer from earlier messages.
- When the customer states a lasting fact (age, favorite genre, etc.), call remember_preference to save it.

Ask for the customer's email before any account lookup.
Never reveal one customer's data to another.
"""


# --- emails --- 
class Context(TypedDict):
    customer_email: str

def _customer_ns() -> tuple:
    """Per-customer memory drawer, keyed by their email (cleaned for the Store)."""
    email = get_config()["configurable"].get("customer_email", "unknown")
    safe = email.replace(".", "_").replace("@", "_")
    return ("preferences", safe)


# --- middleware --- 
@dynamic_prompt
def with_customer(request) -> str:
    email = request.runtime.context.get("customer_email", "unknown")
    return SYSTEM + f"\n\nThe logged-in customer's email is: {email}."

# --- agent ---
tools = [
    search_tracks,
    recommend_by_genre,
    get_my_orders,
    get_my_account,
    remember_preference,
    recall_preferences,
]

graph = create_agent(
    model=llm,
    tools=tools,
    middleware=[with_customer],
    context_schema=Context,
)