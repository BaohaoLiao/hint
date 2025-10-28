#######################################################################
# Generate hint with strongr LLMs
#######################################################################

import os
import json
import argparse
from tqdm import tqdm

import datasets
from pychomsky.chchat import AzureOpenAIChatWrapper
from langchain.schema import HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are a tutoring assistant that generates progressive hints to help students solve difficult problems without revealing the solution directly.

TASK:
Given a question and its solution, generate 3 levels of hints that progressively guide the student toward solving the problem independently.

HINT LEVELS:
- Level 1: Minimal hint - Points to the key concept or approach without specifics
- Level 2: Medium hint - Provides more direction on the method or intermediate steps
- Level 3: Detailed hint - Gives substantial guidance while still requiring the student to complete the solution

GUIDELINES:
- Never reveal the final answer
- Each level should build on the previous one
- Hints should inspire problem-solving, not just provide steps to copy
- Tailor hint difficulty to bridge the gap between the student's level and the solution

OUTPUT FORMAT:
```json
{
    "level_1": "minimal hint text",
    "level_2": "medium hint text",
    "level_3": "detailed hint text"
}
```
"""

USER_PROMPT_TEMPLATE = """Question: 
{problem}

Solution:
{solution}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="CLI Configuration")
    
    # dataset
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="validated.json",
        help="Path to the dataset file"
    )
    
    # model
    parser.add_argument(
        "--model-name",
        type=str,
        default="azure-chat-completions-gpt-5-nano-2025-08-07-sandbox",
        help="Name of the model to use"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        help="Reasoning effort level"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8000,
        help="Maximum number of new tokens to generate"
    )
    
    # Save
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gen_hints",
        help="Directory to save output files"
    )
    
    return parser.parse_args()


def make_hint_prompt(question, solution):
    user_prompt = USER_PROMPT_TEMPLATE.format(
        problem=question,
        solution=solution,
    )
    conv = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=[
            {
                "type": "text",
                "text": user_prompt,
            }
        ]),
    ]
    return conv


def generate_hint(llm, message):
    try:
        response = llm(message).content
    except Exception as e: 
        print(e)
        response = ""
    return response


def main(args):
    # Load and prepare dataset
    ds = datasets.load_dataset("json", data_files=args.dataset_path, split="train")
    samples  = []
    for sample in ds:
        problem = sample["problem"]
        solution = sample["solution"]
        message = make_hint_prompt(problem, solution)

        samples.append({
            "problem": problem,
            "solution": solution,
            "answer": sample["answer"],
            "message": message,
        })

    # Initialize LLM
    llm = AzureOpenAIChatWrapper(
        model_name=args.model_name,
        max_completion_tokens=args.max_new_tokens,
        reasoning_effort=args.reasoning_effort,
        response_format={"type": "json_object"},
    )

    # Generate hints
    for sample in tqdm(samples, desc="Generating hints", total=len(samples)):
        hint = generate_hint(llm, sample["message"])
        sample["hint"] = hint

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "hints.json")
    with open(output_path, "w", encoding="utf8") as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved hints to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)