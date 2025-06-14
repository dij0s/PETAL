import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

import argparse
import time
import json
import re
from functools import reduce

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage
from langchain.prompts import PromptTemplate

from pydantic import BaseModel, Field
from modelling.PydanticStreamOutputParser import PydanticStreamOutputParser

from provider.GraphProvider import GraphProvider, State
from modelling.structured_output import BenchmarkScore

parser = PydanticStreamOutputParser(pydantic_object=BenchmarkScore, diff=True)

class Benchmark:
    """
    Benchmarking framework implementation.

    Evaluates the answers generated by the energy planning decision-helper implementation.
    Each prompt result is evaluated against these benchmarks:
    - G-evaluation score (LLM as a judge)
    - Time taken to generate the answer
    """

    def __init__(self, filename: str):
        """
        Initializes the Benchmark class.

        Args:
            filename (str): The filename for outputting benchmark results.
        """
        self._OUTPUT_FILENAME = filename

        self._queue = asyncio.Queue()
        self._graph_state = None
        self._is_running = False

        MODEL = os.getenv("OLLAMA_MODEL_LLM_BENCHMARKING", "llama3.2:3b")
        self._llm = ChatOllama(model=MODEL, temperature=0).with_structured_output(BenchmarkScore)

        self.REDIS_URL_MEMORIES = os.getenv("REDIS_URL_MEMORIES")
        if self.REDIS_URL_MEMORIES is None:
            raise ValueError("REDIS_URL_MEMORIES environment variable must be set")

        self.THREAD_ID = "1000"
        self.USER_ID = "999"

        self._system_prompt = PromptTemplate.from_template("""
        You are an expert evaluator for municipal energy planning AI systems.

        CONTEXT:
        - Municipality: {location}
        - User Query: {query}
        - Municipal Data Available: {municipal_data}
        - Cantonal Guidelines: {cantonal_guidelines}

        RESPONSE TO EVALUATE:
        {response}

        COMPREHENSIVE EVALUATION:
        Evaluate the response across all four criteria and provide scores for each (1-5 scale):

        1. DATA INTERPRETATION (1-5):
        Rate how accurately the response interprets and presents data:
        - 5: Correctly interprets all data points, understands units, identifies data gaps, presents data clearly
        - 4: Minor interpretation issues but generally accurate
        - 3: Some data misinterpretation but core understanding is correct
        - 2: Significant data interpretation errors
        - 1: Completely misinterprets data or presents false information

        Focus on:
        - Are municipal data values presented correctly without scaling annotations?
        - Are units preserved and clearly stated?
        - Is the distinction between different data types clear?
        - Are calculations (like totals, gaps) accurate?

        2. GUIDELINE APPLICATION (1-5):
        Rate how well the response applies cantonal guidelines to municipal planning:
        - 5: Appropriately applies guidelines as policy framework, scales specific targets correctly, maintains cantonal context
        - 4: Good application with minor issues
        - 3: Generally applies guidelines correctly but with some confusion
        - 2: Misapplies guidelines or confuses their scope
        - 1: Completely misunderstands or ignores guidelines

        Focus on:
        - Are cantonal guidelines used as policy framework rather than direct municipal constraints?
        - Are specific targets scaled appropriately for the municipality?
        - Is the relationship between cantonal and municipal planning clear?

        3. MUNICIPAL RELEVANCE (1-5):
        Rate how relevant and actionable the response is for municipal energy planning:
        - 5: Highly relevant, actionable recommendations, appropriate scope for municipal planning
        - 4: Relevant with minor scope issues
        - 3: Generally relevant but some recommendations may be too broad/narrow
        - 2: Limited relevance to municipal planning needs
        - 1: Not relevant to municipal planning or provides misleading guidance

        Focus on:
        - Are recommendations appropriate for municipal-level decision making?
        - Does it address the specific municipality's context?
        - Are next steps actionable at the municipal level?

        4. SOURCE CITATIONS (1-5):
        Rate the quality and accuracy of source citations:
        - 5: All claims properly cited, correct citation format, distinguishes between data sources and guidelines
        - 4: Most claims cited correctly with minor formatting issues
        - 3: Generally cites sources but with some omissions or format issues
        - 2: Inconsistent or incorrect citations
        - 1: Missing citations or completely incorrect citation format

        Focus on:
        - Are official guidelines cited using the **Source** format?
        - Are municipal data sources handled appropriately (no citation needed)?
        - Is the citation format consistent and correct?

        SCORING INSTRUCTIONS:
        Provide your evaluation in this format:

        {scoring_format}
        """
        )

    async def prompt(self, request: str):
        """Puts the request and on_end callback into the queue."""
        await self._queue.put((request, lambda state, time: self._benchmark(request, state, time)))

    async def await_completion(self):
        """Waits for all pending requests to finish."""
        while not self._queue.empty() or self._is_running:
            await asyncio.sleep(0.1)

    def _consume(self, mode, chunk):
        """Consumes the incoming chunks from the LLM."""
        if mode == "values":
            self._graph_state = State(**chunk)

    def _benchmark(self, request: str, state: State, time: float):
        """Benchmarks the argument-given request."""
        last_ai_message = next(msg.content for msg in reversed(state.messages) if isinstance(msg, AIMessage))
        # remove reasoning (if does)
        # from ai message
        if isinstance(last_ai_message, str):
            last_ai_message = re.sub(r"<think>.*?</think>", "", last_ai_message, flags=re.DOTALL)
        last_ai_message = last_ai_message.strip() # type: ignore

        if state.router is None or state.geocontext is None:
            raise ValueError("Router and Geocontext must be initialized.")
        # prompt judging llm
        prompt = [SystemMessage(content=self._system_prompt.format(
            location=state.router.location,
            query=request,
            municipal_data=state.geocontext.context_tools,
            cantonal_guidelines=state.geocontext.context_constraints,
            response=last_ai_message,
            scoring_format=parser.get_description()
        ))]
        response = self._llm.invoke(prompt)
        if not isinstance(response, BenchmarkScore):
            raise ValueError("Invalid response type")

        self._save_score(response, request, last_ai_message, time) # type: ignore

    def _save_score(self, score: BenchmarkScore, request: str, response: str, time: float):
        output_path = self._OUTPUT_FILENAME
        record = {
            "request": request,
            "response": response,
            "score": score.model_dump(),
            "time": time
        }

        try:
            if os.path.exists(output_path):
                with open(output_path, "r+", encoding="utf-8") as f:
                    try:
                        f.seek(0)
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = []
                    except Exception:
                        data = []
                    data.append(record)
                    f.seek(0)
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.truncate()
            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump([record], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving benchmark record: {e}")

    async def start(self):
        """
        Consumes incoming requests from the queue and processes them.

        At the end, trigger on_end callback.
        """

        async with GraphProvider.build(self.REDIS_URL_MEMORIES) as graph: # type: ignore
            while True:
                try:
                    # consume request
                    request, on_end = await self._queue.get()

                    self._is_running = True
                    start = time.time()

                    await graph.stream_graph_updates(self.THREAD_ID, self.USER_ID, request, self._consume, with_state=True)

                    # execute callback
                    on_end(self._graph_state, time.time() - start)
                    self._is_running = False
                except Exception as e:
                    print(f"Error: {e}")
                    break

def _parse_file(filepath: str) -> list[str]:
    """Load prompts from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        raise e

async def main():
    """Benchmark requests from the CLI."""
    parser = argparse.ArgumentParser(description="Benchmark requests from the CLI.")
    parser.add_argument('--output', '-o', type=str, required=True, help='Output filename')
    # user must provide either
    # a single prompt or a file
    # containing prompts
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt', '-p', type=str, help='Single prompt to benchmark')
    group.add_argument('--file', '-f', type=str, help='File containing prompts (one per line)')


    args = parser.parse_args()

    benchmark = Benchmark(args.output)
    benchmark_task = asyncio.create_task(benchmark.start())

    try:
        prompts = [args.prompt] if args.prompt else _parse_file(args.file)
        for prompt in prompts:
            await benchmark.prompt(prompt)
        # await benchmark finish
        await benchmark.await_completion()
    except Exception as e:
        print(e)
    finally:
        benchmark_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
