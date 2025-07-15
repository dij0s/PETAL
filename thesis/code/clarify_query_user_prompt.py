PromptTemplate.from_template("""
The user just queried some information and you need additional details about:

{needed_information}

User input: "{user_input}"
""")
