"""
Chinook Music Store — Customer Support Bot
"""

from pathlib import Path
import sqlite3

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.config import get_store  # store memory on RAM

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
def get_my_orders(customer_email: str) -> list:
    """Get a customer's purchase history by email. (Security gets fixed in step 3.)"""
    return run_sql(
        """
        SELECT i.InvoiceId, i.InvoiceDate, i.Total
        FROM Invoice i
        JOIN Customer c ON i.CustomerId = c.CustomerId
        WHERE c.Email = ?
        ORDER BY i.InvoiceDate DESC LIMIT 10
        """,
        (customer_email,),
    )

@tool
def get_my_account(customer_email: str) -> dict:
    """Get a customer's account info by email."""
    rows = run_sql(
        "SELECT FirstName, LastName, Email, Country FROM Customer WHERE Email = ?",
        (customer_email,),
    )
    return rows[0] if rows else {"error": "not found"}

@tool
def remember_preference(key: str, fact: str) -> str:
    """Save a lasting fact under a key, e.g. key='age', fact='21'."""
    store = get_store()
    store.put(("preferences",), key, {"fact": fact})
    return f"Saved {key}: {fact}"

@tool
def recall_preferences() -> str:
    """Look up saved facts about the customer from long-term memory."""
    store = get_store()
    item = store.get(("preferences",), "note")
    return item.value["fact"] if item else "No saved preferences yet."


# --- system prompt ---
SYSTEM = """You are a support agent for the Chinook Music Store.
You help with: 1) MUSIC — search and recommend by genre. 2) ACCOUNT — orders and account details.

MEMORY RULES:
- At the start of a conversation, call recall_preferences to load what you know about the customer.
- Whenever the customer states a lasting fact (their age, a favorite genre, etc.), call remember_preference to save it.

Ask for the customer's email before any account lookup.
Never reveal one customer's data to another.
"""
# --- agent (no checkpointer: the dev server provides persistence) ---
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
    system_prompt=SYSTEM,
)