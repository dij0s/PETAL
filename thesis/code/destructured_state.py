class State:
    messages: list
    lang: str

    intent: str
    location: str
    aggregated_query: str
    conversation_type: str
    needs_clarification: bool
    needs_memoization: bool

    context_tools: dict
    context_constraints: list
