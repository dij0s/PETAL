PromptTemplate.from_template("""
**Previous Context:**
- Last input: "{previous_user_input}"
- Current state: {current_router}

**Current Input:** "{user_input}"
""")
