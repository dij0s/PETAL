PromptTemplate.from_template("""
Given that the municipality "{location}" has {residents_count} residents and an exploitable surface of {exploitable_surface} ha, we redacted the following report to answer the user request.

This is the data we have at our disposal:

### Datapoints for "{location}"
{datapoints_description}

### Guidelines for "{location}"
{guidelines}

And this is our answer to the user request "{user_request}":
{llm_answer}

Check for common interpretation errors:
1. **Mathematical Accuracy**: Are all calculations correct? Check arithmetic carefully
2. **Data Type Logic**: Are energy types properly distinguished and not inappropriately aggregated?
3. **Units & Precision**: Are units preserved, consistent, and meaningful?

If the answer is correct, complete and does not contain strong interpretation errors, return retry: False. Otherwise, return retry: True.
""")
