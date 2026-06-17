"""
Chinook Music Store — Customer Support Bot
"""

from pathlib import Path
import sqlite3

from langchain_anthropic import ChatAnthropic
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

from typing_extensions import TypedDict
from langchain.agents.middleware import (
    dynamic_prompt,
    SummarizationMiddleware,
    AgentMiddleware,
    HumanInTheLoopMiddleware,
)
from langchain_core.messages import AIMessage

import re

# --- database help ---
DB_PATH = Path(__file__).parent.parent.parent / "chinook-db" / "chinook.db"

def run_sql(sql: str, params: tuple = ()) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- emails (at runtime)--- 
class Context(TypedDict):
    customer_email: str

def _customer_ns(email: str) -> tuple:
    """Per-customer memory drawer, keyed by their email (cleaned for the Store)."""
    safe = email.replace(".", "_").replace("@", "_")
    return ("preferences", safe)

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
def get_my_orders(runtime: ToolRuntime[Context]) -> list:
    """Get the logged-in customer's purchase history."""
    email = runtime.context["customer_email"]
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
def get_my_account(runtime: ToolRuntime[Context]) -> dict:
    """Get the logged-in customer's account info."""
    email = runtime.context["customer_email"]
    rows = run_sql(
        "SELECT FirstName, LastName, Email, Country FROM Customer WHERE Email = ?",
        (email,),
    )
    return rows[0] if rows else {"error": "not found"}


@tool
def get_order_details(runtime: ToolRuntime[Context]) -> list:
    """Get the logged-in customer's purchases broken down by song, artist, album, and genre."""
    email = runtime.context["customer_email"]
    return run_sql(
        """
        SELECT i.InvoiceId, i.InvoiceDate,
               t.Name AS track, ar.Name AS artist, al.Title AS album,
               g.Name AS genre,
               il.UnitPrice, il.Quantity
        FROM Invoice i
        JOIN Customer c     ON i.CustomerId = c.CustomerId
        JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
        JOIN Track t        ON t.TrackId = il.TrackId
        LEFT JOIN Album al  ON al.AlbumId = t.AlbumId
        LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
        LEFT JOIN Genre g   ON g.GenreId = t.GenreId
        WHERE c.Email = ?
        ORDER BY i.InvoiceDate DESC
        """,
        (email,),
    )

@tool
def count_my_tracks_by_genre(runtime: ToolRuntime[Context]) -> list:
    """Count the logged-in customer's purchased tracks by genre using exact SQL aggregation."""
    email = runtime.context["customer_email"]
    return run_sql(
        """
        SELECT g.Name AS genre, SUM(il.Quantity) AS track_count
        FROM Invoice i
        JOIN Customer c     ON i.CustomerId = c.CustomerId
        JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
        JOIN Track t        ON t.TrackId = il.TrackId
        JOIN Genre g        ON g.GenreId = t.GenreId
        WHERE c.Email = ?
        GROUP BY g.Name
        ORDER BY track_count DESC, g.Name ASC
        """,
        (email,),
    )

@tool
def get_my_support_rep(runtime: ToolRuntime[Context]) -> dict:
    """Get the logged-in customer's assigned support rep and their contact info."""
    email = runtime.context["customer_email"]
    rows = run_sql(
        """
        SELECT e.FirstName, e.LastName, e.Title, e.Email, e.Phone
        FROM Customer c
        JOIN Employee e ON c.SupportRepId = e.EmployeeId
        WHERE c.Email = ?
        """,
        (email,),
    )
    return rows[0] if rows else {"error": "No support rep assigned."}

@tool
def remember_preference(key: str, fact: str, runtime: ToolRuntime[Context]) -> str:
    """Save a lasting fact about THIS customer, e.g. key='genre', fact='jazz'."""
    store = runtime.store
    email = runtime.context["customer_email"]
    store.put(_customer_ns(email), key, {"fact": fact})
    return f"Saved {key}: {fact}"

@tool
def recall_preferences(runtime: ToolRuntime[Context]) -> str:
    """Load everything saved about THIS customer."""
    store = runtime.store
    email = runtime.context["customer_email"]
    items = store.search(_customer_ns(email))
    if not items:
        return "No saved preferences yet."
    return "; ".join(f"{it.key}: {it.value.get('fact', it.value)}" for it in items)

@tool
def process_refund(invoice_id: int, reason: str, runtime: ToolRuntime[Context]) -> str:
    """Process a refund for one of the logged-in customer's invoices. Requires human approval."""
    email = runtime.context["customer_email"]
    # security: confirm the invoice belongs to THIS customer
    rows = run_sql(
        """
        SELECT i.InvoiceId, i.Total
        FROM Invoice i JOIN Customer c ON i.CustomerId = c.CustomerId
        WHERE i.InvoiceId = ? AND c.Email = ?
        """,
        (invoice_id, email),
    )
    if not rows:
        return f"Invoice {invoice_id} not found for this customer."
    return f"Refund approved for invoice {invoice_id} (${rows[0]['Total']}). Reason: {reason}."

# --- system prompt ---
SYSTEM = """You are a support agent for the Chinook Music Store.

GREETING:
- On your VERY FIRST reply in a conversation, begin with this one-line welcome, then immediately answer whatever the customer asked:
  "Hi! I'm the Chinook Music Store support assistant — happy to help with music or your account."
- Do this only once, on the first reply. Never greet again later in the same conversation.

You help with:
1) MUSIC — search tracks and recommend by genre.
2) ACCOUNT — order history, detailed purchases (songs and artists per order), account details, and the customer's assigned support rep's contact info.

MEMORY RULES:
- You maintain the customer's preference profile automatically. The customer NEVER needs to ask you to save anything.
- NEVER ask permission to save (e.g. "would you like me to save this?"). Just save it, then briefly mention you've noted it.
- ALWAYS call recall_preferences at the start to load what you know. Answer about saved preferences ONLY from its result.
- Whenever you learn something durable about the customer — their age, favorite genre, favored artists, preferred format — whether they state it OR you infer it confidently from purchases or conversation, immediately call remember_preference to save it. Do this WITHOUT being asked.
- After saving, add a short note like "I've saved that you like rock." Keep it to one line.
- Only save durable facts (tastes, age, preferences). Never save one-off questions.

COUNTING RULES:
- For questions asking "how many", "most common genre", "count", or "breakdown by genre", use count_my_tracks_by_genre instead of manually counting rows from get_order_details.

REFUNDS:
- To process any refund you MUST call the process_refund tool with the invoice_id and reason.
- Never tell the customer a refund was submitted unless you actually called process_refund.

IDENTITY AND PRIVACY:
- The customer is already identified from context — never ask for their email; account tools use it automatically.
- All account tools ONLY ever return the logged-in customer's own data. You cannot look up anyone else.
- If the customer asks about another person's account, orders, or purchases BY NAME or email (e.g. "show me Jane Smith's orders", "what did bob@x.com buy"), DO NOT answer with anyone's data. Briefly explain you can only access their own account, then offer to help with their own records instead.
- Never present the logged-in customer's data as if it belonged to the other person they named.
"""

# --- middleware --- 
# email is in context, at runtime
@dynamic_prompt
def with_customer(request) -> str:
    ctx = request.runtime.context or {}        # fall back to empty dict if None
    email = ctx.get("customer_email", "unknown")
    return SYSTEM + f"\n\nThe logged-in customer's email is: {email}."

# summarization: compress history once a thread gets large
summarization = SummarizationMiddleware(
    model="claude-sonnet-4-5",
    trigger=("tokens", 5000),   # fire only when the conversation exceeds 5k tokens
    keep=("messages", 5),         # always keep the 5 most recent messages in full
)

# SQL-injection tripwire: block obvious DB-attack input
INJECTION_PATTERNS = (
    "drop table", "delete from", "insert into", "update ",
    "union select", "' or '1'='1", "or 1=1", ";--", "--", "/*", "xp_",
)

class SQLInjectionGuard(AgentMiddleware):
    """Defense-in-depth: parameterized queries are the real protection;
    this flags and blocks obvious injection attempts and makes them visible in traces."""
    def before_model(self, state, runtime):
        msgs = state.get("messages", [])
        last = msgs[-1].content.lower() if msgs else ""
        if any(p in last for p in INJECTION_PATTERNS):
            return {
                "messages": [AIMessage(
                    "That request looks like it contains a database command, which I can't run. "
                    "I can help you search music or look up your account — what would you like?"
                )],
                "jump_to": "end",
            }
        return None

class OtherCustomerGuard(AgentMiddleware):
    """Hard block: if the user asks for another person's data by name or email,
    refuse before the model runs. The hardened tools already prevent leakage;
    this makes the *refusal behavior* deterministic instead of prompt-dependent."""

    # email, possessive name, "orders for First Last", or "what did First Last buy"
    # - "show me Helena's orders"
    # - "show me Helena Holy's orders"
    # - "show me orders for Helena Holy"
    NAME_POSSESSIVE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:'s|s')\b")
    NAME_REFERENCE = re.compile(
        r"\b(?:for|by|about|from)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
    )
    NAMED_BUYER = re.compile(
        r"\bwhat\s+did\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:buy|purchase|order)\b",
        re.IGNORECASE,
    )
    EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    ACCOUNT_WORDS = (
        "order", "orders", "purchase", "purchases", "account",
        "spent", "invoice", "invoices", "bought", "buy", "data", "history"
    )

    def before_model(self, state, runtime):
        msgs = state.get("messages", [])
        if not msgs:
            return None
        last = msgs[-1].content
        if not isinstance(last, str):
            return None
        lower = last.lower()

        asks_account = any(w in lower for w in self.ACCOUNT_WORDS)
        if not asks_account:
            return None

        # email belonging to someone, OR a Name's possessive
        if (
            self.EMAIL.search(last)
            or self.NAME_POSSESSIVE.search(last)
            or self.NAME_REFERENCE.search(last)
            or self.NAMED_BUYER.search(last)
        ):
            return {
                "messages": [AIMessage(
                    "For privacy, I can only access your own account — I can't look up "
                    "another customer's orders or details. I'm happy to help with your "
                    "own orders, purchases, or support rep. What would you like?"
                )],
                "jump_to": "end",
            }
        return None


# human-in-the-loop: pause refunds for human approval
hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "process_refund": {"allowed_decisions": ["approve", "reject"]},
    },
    description_prefix="Refund pending human approval",
)

# --- agent ---
tools = [
    search_tracks,
    recommend_by_genre,
    get_my_orders,
    get_order_details,
    count_my_tracks_by_genre,
    get_my_account,
    get_my_support_rep,
    process_refund,
    remember_preference,
    recall_preferences,
]

graph = create_agent(
    model=llm,
    tools=tools,
    middleware=[
        SQLInjectionGuard(),
        OtherCustomerGuard(),
        with_customer,
        hitl,
        summarization,
    ],
    context_schema=Context,
)