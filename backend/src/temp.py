import os
import asyncio

import argparse
import time
import json
import re
from functools import reduce

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage
from langchain.prompts import PromptTemplate

from provider.GraphProvider import GraphProvider, State
from pydantic import BaseModel, Field
from modelling.PydanticStreamOutputParser import PydanticStreamOutputParser
from modelling.structured_output import BenchmarkScore, RouterOutput

parser = PydanticStreamOutputParser(pydantic_object=BenchmarkScore, diff=True)
llm = (
    ChatOllama(model="deepseek-r1:8b", temperature=0)
        .with_structured_output(BenchmarkScore)
)
system_prompt = PromptTemplate.from_template("""
You are an expert evaluator for municipal energy planning AI systems. You must be CRITICAL and SKEPTICAL - good formatting and professional language do not equal good energy planning advice.

CONTEXT:
- Municipality: {location}
- User Query: {query}
- Query Type: {query_type} (factual = data analysis focus, actionable = planning methodology focus)

RESPONSE TO EVALUATE:
{response}

CRITICAL EVALUATION FRAMEWORK:
Rate each criterion from 1-5, where 5 requires EXCEPTIONAL quality, not just "looks professional":

1. DATA INTERPRETATION (1-5):
CRITICAL CHECKS - Be harsh on these:
- **Query Relevance**: Does it ONLY use data that directly addresses the user's specific question?
- **Mathematical Accuracy**: Are all calculations correct? Check arithmetic carefully
- **Data Type Logic**: Are energy types properly distinguished and not inappropriately aggregated?
- **Units & Precision**: Are units preserved, consistent, and meaningful?
- **Zero Value Handling**: Are zero values correctly interpreted as "complete absence" not "constraints"?

DATA SANITY CHECKS - Common failure modes from expert feedback:
- **CONSUMPTION vs PRODUCTION**: Does it mix these without proper justification?
- **DOUBLE-COUNTING**: Could the same energy be counted twice? (e.g., district heating production + end consumption)
- **INVALID AGGREGATION**: Are summed values actually additive? (Don't add PV + solar thermal inappropriately)
- **IRRELEVANT DATA INCLUSION**: Does it include retrieved data that doesn't match query focus?
- **TEMPORAL LOGIC**: Do timeframes make sense? (No past-year projections, consistent time references)
- **PRODUCTION vs POTENTIAL**: Are actual vs theoretical values properly distinguished?
- **DATA AVAILABILITY CLAIMS**: Does it claim "no data" while having access to that information?

RED FLAGS (automatic score ≤2):
- Using irrelevant data just because it was retrieved
- Mixing consumption/production inappropriately
- Double-counting energy flows
- Invalid aggregation of incompatible data types
- Temporal inconsistencies (projecting past years)
- Mathematical errors in calculations

2. METHODOLOGY ALIGNMENT (1-5):
For FACTUAL queries - Data Analysis Standards:
- **Current Energy Profile**: Clear presentation of key data points, trends, patterns
- **Data Insights**: Identifies significant findings and their implications
- **Factual Focus**: Avoids planning recommendations, stays analytical

For ACTIONABLE queries - Planning Methodology Standards:
- **Structured Approach**: Follows resource assessment → optimization strategy → implementation roadmap
- **Guideline Integration**: Properly extracts timeframes, implementation phases, milestones
- **Collaborative Guidance**: Provides next planning steps and local validation questions
- **Expert Positioning**: Presents as collaborative advisor, not directive authority

RED FLAGS (automatic score ≤2):
- Wrong methodology for query type (planning advice for factual query, or pure data for actionable query)
- Generic advice that ignores municipal context
- Missing key methodology components for query type

3. MUNICIPAL RELEVANCE (1-5):
CRITICAL CHECKS - This is where most responses fail:
- **Specificity**: Are insights/recommendations specific to THIS municipality, not generic?
- **Scale Appropriateness**: Are suggestions feasible at municipal scale and authority?
- **Query Alignment**: Does it directly address the specific question asked?
- **Local Context**: Does it consider municipal-specific constraints and opportunities?
- **Implementation Feasibility**: Are next steps actionable by municipal decision-makers?

RED FLAGS (automatic score ≤2):
- Generic recommendations applicable to any municipality
- Suggesting actions outside municipal authority
- Not answering the specific question asked
- Ignoring obvious municipal constraints or context

4. TECHNICAL COMPLIANCE (1-5):
FORMAT REQUIREMENTS:
- **Language Consistency**: Response in correct language throughout
- **Header Structure**: Uses only H3 (###) and H4 (####) headers
- **Citation Format**: Official sources cited as **Source Name** format
- **Structure Quality**: Appropriate use of tables, bullets, formatting

CONTENT REQUIREMENTS:
- **Source Accuracy**: Citations actually support the claims made
- **Professional Presentation**: Hides internal tool names, maintains expert positioning
- **Completeness**: Includes required conclusion sections (next steps, recommendations)

RED FLAGS (automatic score ≤2):
- Incorrect citation format or false citations
- Wrong header levels or poor formatting
- Missing required sections for query type
- Language inconsistencies

SCORING PHILOSOPHY:
- Score 1-2: Major errors, misleading advice, fundamental misunderstanding, or wrong methodology
- Score 3: Basic competence but significant limitations, generic advice, or methodology gaps
- Score 4: Good quality with minor issues, genuinely useful for municipal planning, correct methodology
- Score 5: Exceptional - accurate, specific, actionable, demonstrates deep understanding, perfect methodology alignment

BE ESPECIALLY CRITICAL OF:
- Responses that include irrelevant data just because it was retrieved
- Professional-sounding advice that's actually generic or inappropriate
- Methodology mismatches (planning advice for data queries, or vice versa)
- Mathematical errors disguised by confident presentation
- Claims not supported by actual data or guidelines

SCORING FORMAT:
{scoring_format}

Remember: Your job is to catch responses that sound authoritative but provide poor energy planning guidance. Be especially harsh on data misuse and methodology misalignment.
""")

def score(request, ai_message, router) -> BenchmarkScore:
    prompt = [SystemMessage(content=system_prompt.format(
        location=router.location,
        query=request,
        query_type=router.intent,
        response=ai_message,
        scoring_format=parser.get_description()
    ))]
    response = llm.invoke(prompt)
    if not isinstance(response, BenchmarkScore):
        raise ValueError("Invalid response type")
    return response

for index in range(10):
    with open(f"benchmarking/sion_small_benchmark_0{index}.json", "r") as file:
        data = json.load(file)

    results = reduce(
        lambda res, x: res + [score(
            x.get("request"),
            x.get("response"),
            RouterOutput(**x.get("router"))
        ).model_dump()],
        data.get("results", []),
        []
    )
    with open(f"benchmarking/small_benchmark_dp_0{index}.json", "w") as out:
        json.dump(results, out, indent=2)
