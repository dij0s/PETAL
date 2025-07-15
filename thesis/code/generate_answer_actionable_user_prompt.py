PromptTemplate.from_template("""
## Energy Planning Guidance for {location}

### Available Data
{tools_data}

### Official Guidelines & Regulations
{constraints}

### User Request
**Query Type**: Actionable (requires planning methodology and policy guidance)
**Focus**: {aggregated_query}
**Original**: {user_query}

### Applied User Preferences
{memories_description}

---
**TASK**: Guide the user through systematic energy planning methodology, combining data analysis with regulatory compliance and local expertise integration. Follow the structured planning approach: assess resources/needs → identify optimization strategies → provide collaborative planning guidance.
""")
