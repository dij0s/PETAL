import re
import requests
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from provider.ModelProvider import ModelProvider

scale = []
no_scale = []

@dataclass
class EnergyNumber:
    """Represents an energy number found in text with its context"""
    number: float
    unit: str
    full_match: str  # e.g., "500 MW"
    context_before: str
    context_after: str
    start_pos: int
    end_pos: int
    classification: Optional[str] = None

class EnergyNumberProcessor:
    def __init__(self, ollama_base_url: str = "http://localhost:11434", model_name: str = "llama3.2:3b"):
        self.ollama_base_url = ollama_base_url

        self.model = ModelProvider.from_env_variable("bullshit", temperature=0, defaults=model_name, top_p=0.1, num_predict=5, extract_reasoning=True)

        # Regex pattern for energy numbers
        self.energy_pattern = re.compile(
            r'\b(\d+(?:[\.,]\d+)?)\s+(GWh/hivernale|GWh/month|GWh/year|GWh/a|GWhel|GWhth|GWh|CHF)\b(?!\s+per\b)',
            re.IGNORECASE
        )

        self.classification_prompt = """Look at this text and decide if the energy number should be scaled down for a smaller municipality.

Text: "{context}"

Is this number a quantity that should be scaled? Answer ONLY with YES or NO.

YES if it's: capacity, generation amount, consumption, storage, demand
NO if it's: efficiency %, voltage, frequency, technical spec, threshold

Answer:"""

    def extract_energy_numbers(self, text: str, context_window: int = 200) -> List[EnergyNumber]:
        """Extract energy numbers with surrounding context"""
        energy_numbers = []

        for match in self.energy_pattern.finditer(text):
            print(match)
            number_str = match.group(1).replace(',', '')  # Remove commas
            try:
                number = float(number_str)
            except ValueError:
                continue

            unit = match.group(2)
            full_match = match.group(0)
            start_pos = match.start()
            end_pos = match.end()

            # Extract context
            context_start = max(0, start_pos - context_window)
            context_end = min(len(text), end_pos + context_window)

            context_before = text[context_start:start_pos].strip()
            context_after = text[end_pos:context_end].strip()

            energy_numbers.append(EnergyNumber(
                number=number,
                unit=unit,
                full_match=full_match,
                context_before=context_before,
                context_after=context_after,
                start_pos=start_pos,
                end_pos=end_pos
            ))

        return energy_numbers

    def classify_number(self, energy_num: EnergyNumber) -> str:
        """Classify a single energy number using Ollama"""
        context = f"{energy_num.context_before} {energy_num.full_match} {energy_num.context_after}"
        print(f"context ici mgl: {context}")
        prompt = self.classification_prompt.format(context=context.strip())

        try:
            response = self.model.invoke(prompt)
            response_text = response.content

            print(f"Raw response: '{response_text}'")  # Debug output

            # Simple classification based on YES/NO
            if 'YES' in response_text:
                scale.append(energy_num.full_match)
                return 'SCALE'
            elif 'NO' in response_text:
                no_scale.append(energy_num.full_match)
                return 'KEEP'
            else:
                return 'UNCERTAIN'

        except Exception as e:
            print(f"Error classifying number: {e}")
            return 'UNCERTAIN'

    def process_document(self, text: str, scaling_factor: float) -> str:
        """Process a document by extracting, classifying, and scaling energy numbers"""

        # Step 1: Extract all energy numbers
        energy_numbers = self.extract_energy_numbers(text)

        if not energy_numbers:
            return text

        print(f"Found {len(energy_numbers)} energy numbers")

        # Step 2: Classify each number (this is the fast part)
        for energy_num in energy_numbers:
            classification = self.classify_number(energy_num)
            energy_num.classification = classification
            print(f"  {energy_num.full_match} → {classification}")

        # Step 3: Apply scaling (work backwards to preserve positions)
        modified_text = text
        for energy_num in reversed(energy_numbers):  # Reverse to maintain positions
            if energy_num.classification == 'SCALE':
                # Scale the number
                scaled_number = energy_num.number * scaling_factor
                scaled_str = f"{scaled_number:.1f}"

                # Create the replacement
                new_match = f"{scaled_str} {energy_num.unit}"

                # Replace in text
                modified_text = (
                    modified_text[:energy_num.start_pos] +
                    new_match +
                    modified_text[energy_num.end_pos:]
                )

        return modified_text

    def process_multiple_documents(self, documents: List[str], scaling_factor: float) -> List[str]:
        """Process multiple documents"""
        results = []
        for i, doc in enumerate(documents):
            print(f"\nProcessing document {i+1}/{len(documents)}")
            processed_doc = self.process_document(doc, scaling_factor)
            results.append(processed_doc)
        return results

# Usage example
def main():
    # Initialize the processor
    processor = EnergyNumberProcessor()

    # Test documents
    with open("./marimo/documents_with_visual_analysis.json", "r") as file:
        data = json.load(file)
    test_docs = [
        doc["visual_analysis"]["analysis"]
        for doc in data.values()
        ][:100]

    # Process with scaling factor (e.g., 0.1 for a municipality that's 1/10th the size)
    scaling_factor = 0.1

    processed_docs = processor.process_multiple_documents(test_docs, scaling_factor)

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)

    for i, (original, processed) in enumerate(zip(test_docs, processed_docs)):
        print(f"\nDocument {i+1}:")
        print("ORIGINAL:", original)
        print("PROCESSED:", processed)
        print("-" * 40)

if __name__ == "__main__":
    main()
    print(f"SCALING: {scale}")
    print(f"NOT SCALING: {no_scale}")
