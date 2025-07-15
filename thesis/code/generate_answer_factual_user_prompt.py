PromptTemplate.from_template("""
## Energy Data Analysis for {location}

### Available Data
{tools_data}

### User Request
**Query Type**: Factual (data analysis focus)
**Focus**: {aggregated_query}
**Original**: {user_query}

### Applied User Preferences
{memories_description}

---
**TASK**: Provide comprehensive data analysis with insights, trends, and factual findings.
""")
