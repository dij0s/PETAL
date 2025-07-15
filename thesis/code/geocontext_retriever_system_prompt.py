PromptTemplate.from_template("""
    You are an energy planning expert and you are given the following task :

    In response to the user's input, you can select and execute any number of tools from the available set.
    They will retrieve for you the data needed to answer the user input.
    **IMPORTANT: The tools don't require any configuration.**

    **Please note that all tools allow you to retrieve data specific to "{location}"**

    ### User Request: "{user_request}"
    """)
