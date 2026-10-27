from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from datawiz.chat import answer
from datawiz.profile import sample_dataset
from datawiz.retrieval import DataIndex


class ScriptedModel:
    """Stands in for Gemini: returns the given replies in order and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def bind_tools(self, tools):
        self.tool_names = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def tool_call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_tool_results_reach_the_model():
    df = sample_dataset()
    model = ScriptedModel([
        tool_call("aggregate", {"column": "revenue", "agg": "sum", "group_by": ["region"]}),
        AIMessage(content="North has the most revenue."),
    ])
    result = answer("Which region has the highest total revenue?", df, DataIndex(df), model)

    assert result.text == "North has the most revenue."
    assert result.steps[0]["tool"] == "aggregate"
    tool_message = model.calls[1][-1]
    assert isinstance(tool_message, ToolMessage) and "sum_revenue" in tool_message.content
    assert "column revenue" in model.calls[0][0].content  # retrieved profile is in the prompt
    assert set(model.tool_names) == {"aggregate", "filter_rows", "value_counts", "correlation"}


def test_arguments_that_break_the_schema_come_back_as_errors():
    df = sample_dataset()
    model = ScriptedModel([
        tool_call("aggregate", {"column": "revenue", "agg": "average"}),
        AIMessage(content="Sorry, retrying."),
    ])
    result = answer("average revenue?", df, DataIndex(df), model)
    assert "error" in result.steps[0]["result"]


def test_history_is_sent_with_the_question():
    df = sample_dataset()
    model = ScriptedModel([AIMessage(content="ok")])
    history = [{"role": "user", "content": "Look at the West region"}, {"role": "assistant", "content": "Sure."}]
    answer("And its best product?", df, DataIndex(df), model, history)
    sent = model.calls[0]
    assert isinstance(sent[1], HumanMessage) and sent[1].content == "Look at the West region"
    assert sent[-1].content == "And its best product?"


def test_stops_calling_tools_after_the_limit():
    df = sample_dataset()
    loop = [tool_call("value_counts", {"column": "region"}, f"call_{i}") for i in range(2)]
    model = ScriptedModel(loop + [AIMessage(content="Final answer.")])
    result = answer("regions?", df, DataIndex(df), model, max_rounds=2)
    assert result.text == "Final answer." and len(result.steps) == 2
