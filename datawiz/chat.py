"""Answer a question about a DataFrame: retrieve context, let the model call pandas tools, return the answer."""

import json
import os
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .profile import dataset_summary, profile_text
from .retrieval import DataIndex, Retrieved, row_text
from .tools import make_tools

DEFAULT_MODEL = "gemini-flash-latest"
MAX_TOOL_ROUNDS = 4
HISTORY_TURNS = 6
TOOL_RESULT_CHARS = 6000

SYSTEM_PROMPT = """You are DataWiz, a careful data analyst. You answer questions about one table the user uploaded.

Rules:
- Every number in your answer must come from a tool result or from the context below. Never estimate or invent numbers.
- For totals, averages, counts, rankings, comparisons or trends, call a tool. The sample rows below are only examples.
- Use the exact column names from the context. If a question can't be answered from this table, say so and say which data would be needed.
- Keep answers short. Use a markdown table when listing more than three values.

{context}"""


@dataclass
class ChatResult:
    text: str
    steps: list = field(default_factory=list)
    retrieved: Retrieved = field(default_factory=Retrieved)


def get_api_key(secrets=None) -> str | None:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key and secrets is not None:
        try:
            key = secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY")
        except Exception:  # no secrets file
            key = None
    return key


def make_llm(api_key: str, model: str | None = None):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
                                  temperature=0.2, google_api_key=api_key)


def build_context(df: pd.DataFrame, retrieved: Retrieved) -> str:
    parts = ["Dataset:", dataset_summary(df)]
    if retrieved.columns:
        parts += ["", "Columns most relevant to the question:"]
        parts += [f"- {profile_text(p)}" for p in retrieved.columns]
    if not retrieved.rows.empty:
        parts += ["", "Rows that best match the question (examples, not the whole table):"]
        parts += [f"- {row_text(r)}" for _, r in retrieved.rows.iterrows()]
    return "\n".join(parts)


def text_of(message) -> str:
    return str(message.text)


def answer(question: str, df: pd.DataFrame, index: DataIndex, llm, history=(), max_rounds=MAX_TOOL_ROUNDS) -> ChatResult:
    retrieved = index.search(question)
    messages = [SystemMessage(SYSTEM_PROMPT.format(context=build_context(df, retrieved)))]
    for turn in list(history)[-HISTORY_TURNS:]:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(turn["content"]))
    messages.append(HumanMessage(question))

    tools = {t.name: t for t in make_tools(df)}
    model = llm.bind_tools(list(tools.values()))
    steps = []
    for _ in range(max_rounds):
        reply = model.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return ChatResult(text_of(reply), steps, retrieved)
        for call in reply.tool_calls:
            tool = tools.get(call["name"])
            try:
                result = tool.invoke(call["args"]) if tool else {"error": f"No tool named {call['name']}"}
            except (ValueError, TypeError) as e:  # arguments that don't fit the tool's schema
                result = {"error": str(e)}
            steps.append({"tool": call["name"], "args": call["args"], "result": result})
            messages.append(ToolMessage(json.dumps(result, default=str)[:TOOL_RESULT_CHARS], tool_call_id=call["id"]))

    messages.append(HumanMessage("Answer the original question now, using only the tool results above."))
    return ChatResult(text_of(llm.invoke(messages)), steps, retrieved)
