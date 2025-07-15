PromptTemplate.from_template("""
    You are a text processor. Your job is to scale energy numbers in the following documents.

    Instructions:
    - For each document, find energy-related numbers.
    - Multiply ONLY those numbers by {scaling_factor} and replace them in the text, rounded to 1 decimal place.
    - DO NOT scale percentages, dates, or any other numbers.
    - DO NOT add any explanations, notes, or comments.
    - DO NOT change any other part of the text.
    - Return the processed documents, separated by <doc>.

    Input documents:
    {constraints}

    Output:
    Return the same documents, in the same order, separated by <doc>. Only the relevant energy numbers should be changed.
    """)
