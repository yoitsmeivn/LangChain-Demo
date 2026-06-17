"""persist_demo.py — saves every conversation to a SQLite file on disk.
Run as a plain script (NOT through langgraph dev), so the SQLite checkpointer takes effect.
"""

from dotenv import load_dotenv
load_dotenv()

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from agent.graph import llm, tools, SYSTEM  



# 1. a SQLite file on disk — THIS is the saved memory
conn = sqlite3.connect("conversations.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# 2. same agent, now with the file-backed checkpointer
agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM, checkpointer=checkpointer)

# 3. a conversation id — same id = same remembered conversation
config = {"configurable": {"thread_id": "demo-1"}}

while True:
    msg = input("\nYou: ")
    if msg.strip() in {"exit", "quit"}:
        break
    if not msg.strip():        # ignore empty Enter
        continue
    result = agent.invoke({"messages": [{"role": "user", "content": msg}]}, config)
    print("Bot:", result["messages"][-1].content)
