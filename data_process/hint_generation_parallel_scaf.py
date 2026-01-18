#######################################################################
# Generate hint with strongr LLMs
#######################################################################

import os
import json
import argparse
import asyncio
from typing import List, Dict, Any
from tqdm.asyncio import tqdm as async_tqdm

import datasets
from pychomsky.chchat import AzureOpenAIChatWrapper
from langchain.schema import HumanMessage, SystemMessage


SYSTEM_PROMPT = """**[ROLE & GOAL]**
You are an expert AI assistant specializing in problem-solving methodology and knowledge engineering. Your task is to analyze a given problem and its ground-truth solution, and then generate a structured breakdown of the reasoning process with a high degree of granularity.

**[INPUT]**

I will provide you with a "Problem" and its "Ground-Truth Solution".

**[INSTRUCTIONS]**
Based on the provided input, you must generate exactly THREE components. For each component, you MUST generate **a minimum of 4 numbered items**.

- If a natural breakdown results in fewer than 4 items, you must **subdivide the existing steps into more detailed, finer-grained sub-steps** to meet the requirement. For example, a single calculation step can be broken down into 'substituting values', 'performing the operation', and 'stating the result'.

1. **Planning Skeleton**: Extract a high-level planning skeleton. This should be a concise, ordered list of the key reasoning steps and the overall strategy used to reach the solution. Do not include detailed calculations, just the logical flow. Break it down into at least 4 detailed steps.
2. **Knowledge Components**: Identify at least 4 essential knowledge components (like facts, definitions, theorems, lemmas, or formulas) required to solve the problem. List each component clearly in a numbered list.
3. **Solution Breakdown**: Divide the original Ground-Truth Solution into a numbered list of semantically coherent steps or chunks.There should be at least 4 steps or chunks. Each item in the list should be a direct quote or a faithful summary of a part of the original solution text.

**[OUTPUT FORMAT]**
You MUST provide your response in the following structured format. Ensure each section contains at least 4 items.
```json
{
    "PLANNING_SKELETON": "1. [item1]\n...\n4. [item4]\n... (and more if applicable)",
    "KNOWLEDGE_COMPONENTS": "1. [item1]\n...\n4. [item4]\n... (and more if applicable)",
    "SOLUTION_BREAKDOWN": "1. [item1]\n...\n4. [item4]\n... (and more if applicable)"
}
```

**[EXAMPLE]**

--- BEGIN EXAMPLE ---
# ... (Example Problem and Solution are the same)
--- END EXAMPLE ---

**[EXPECTED OUTPUT FOR THE EXAMPLE]**
(This example now demonstrates the required granularity with >= 4 items per section)

```json
{
    "PLANNING_SKELETON": "1. Identify the geometric shape (right-angled triangle) and the goal (find the hypotenuse).\n2. Recall the relevant mathematical theorem that connects the sides of a right-angled triangle.\n3. Formulate the equation by substituting the given side lengths (base and height) into the theorem.\n4. Execute the arithmetic calculation to find the square of the hypotenuse.\n5. Perform the final step of taking the square root to isolate the length of the hypotenuse.",
    "KNOWLEDGE_COMPONENTS": "1. Theorem: Pythagorean Theorem (a² + b² = c² for a right-angled triangle).\n2. Definition: Hypotenuse (The longest side of a right-angled triangle, opposite the right angle).\n3. Concept: Right-angled Triangle (A triangle with one angle measuring 90 degrees).\n4. Mathematical Operation: Square Root (The inverse operation of squaring a number).",
    "SOLUTION_BREAKDOWN": "1. To find the hypotenuse of a right-angled triangle, we can use the Pythagorean theorem, which states that a² + b² = c², where a and b are the lengths of the two shorter sides (legs) and c is the length of the hypotenuse.\n2. The given values are a = 4 cm and b = 3 cm.\n3. Substituting these into the formula gives: 4² + 3² = 16 + 9 = 25.\n4. This means c² = 25. Taking the square root of both sides results in c = 5 cm."
}
```
"""

USER_PROMPT_TEMPLATE = """**[TASK]**
Now, please process the following problem and solution, strictly following all instructions.

**Problem**:
{problem}

**Ground-Truth Solution**:
{solution}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="CLI Configuration")

    # dataset
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="validated.json",
        help="Path to the dataset file",
    )

    # model
    parser.add_argument(
        "--model-name",
        type=str,
        default="azure-chat-completions-gpt-5-nano-2025-08-07-sandbox",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--reasoning-effort", type=str, default="low", help="Reasoning effort level"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16000,
        help="Maximum number of new tokens to generate",
    )

    # Parallelization
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum number of concurrent API calls",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed requests",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        help="Delay in seconds between retries",
    )

    # Save
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gen_hints",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N samples (0 to disable)",
    )

    return parser.parse_args()


def make_hint_prompt(question: str, solution: str) -> List:
    """Create prompt messages for hint generation."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    if not solution or not solution.strip():
        raise ValueError("Solution cannot be empty")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        problem=question,
        solution=solution,
    )
    conv = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": user_prompt,
                }
            ]
        ),
    ]
    return conv


async def generate_hint_async(
    llm: AzureOpenAIChatWrapper,
    message: List,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Dict[str, Any]:
    """
    Generate hint with retry logic and proper error handling.

    Returns a dict with 'success', 'hint', and 'error' keys.
    """
    for attempt in range(max_retries):
        try:
            # Note: If AzureOpenAIChatWrapper doesn't support async natively,
            # we use asyncio.to_thread to run it in a thread pool
            response = await asyncio.to_thread(llm, message)
            hint = response.content

            if not hint or not hint.strip():
                raise ValueError("Empty response from LLM")
            
            # Validate JSON parsing
            try:
                hint_data = json.loads(hint)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {str(e)}")
            
            # Validate all required levels are present
            required_levels = ["PLANNING_SKELETON", "KNOWLEDGE_COMPONENTS", "SOLUTION_BREAKDOWN"]
            missing_levels = [level for level in required_levels if level not in hint_data]

            if missing_levels:
                raise ValueError(f"Missing required hint levels: {', '.join(missing_levels)}")

            # Validate that all levels have non-empty content
            empty_levels = [level for level in required_levels if not hint_data[level] or not str(hint_data[level]).strip()]

            if empty_levels:
                raise ValueError(f"Empty content in hint levels: {', '.join(empty_levels)}")

            return {"success": True, "hint": hint, "error": None}

        except Exception as e:
            error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"

            if attempt < max_retries - 1:
                print(f"{error_msg}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                print(f"{error_msg}. Max retries reached.")
                return {"success": False, "hint": "", "error": str(e)}


async def process_sample(
    llm: AzureOpenAIChatWrapper,
    sample: Dict[str, Any],
    max_retries: int,
    retry_delay: float,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Process a single sample with rate limiting."""
    async with semaphore:
        result = await generate_hint_async(
            llm, sample["message"], max_retries=max_retries, retry_delay=retry_delay
        )

        return {
            **sample,
            "hint": result["hint"],
            "generation_success": result["success"],
            "generation_error": result["error"],
        }


def save_results(samples: List[Dict[str, Any]], output_path: str):
    """Save results to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf8") as f:
        for sample in samples:
            # Remove message object before saving (not JSON serializable)
            sample_to_save = {k: v for k, v in sample.items() if k != "message"}
            json.dump(sample_to_save, f, ensure_ascii=False)
            f.write("\n")


async def main_async(args):
    """Main async execution function."""

    # Load and prepare dataset
    print(f"Loading dataset from {args.dataset_path}...")

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    try:
        ds = datasets.load_dataset("json", data_files=args.dataset_path, split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}")

    if len(ds) == 0:
        raise ValueError("Dataset is empty")

    print(f"Loaded {len(ds)} samples")

    # Prepare samples
    samples = []
    skipped = 0

    for idx, sample in enumerate(ds):
        try:
            problem = sample.get("problem", "")
            solution = sample.get("solution", "")
            answer = sample.get("answer", "")

            if not problem or not solution:
                print(f"Warning: Skipping sample {idx} - missing problem or solution")
                skipped += 1
                continue

            message = make_hint_prompt(problem, solution)

            samples.append(
                {
                    "id": idx,
                    "problem": problem,
                    "solution": solution,
                    "answer": answer,
                    "message": message,
                }
            )

        except Exception as e:
            print(f"Warning: Error preparing sample {idx}: {str(e)}")
            skipped += 1
            continue

    if skipped > 0:
        print(f"Skipped {skipped} invalid samples")

    if not samples:
        raise ValueError("No valid samples to process")

    # Initialize LLM
    print(f"Initializing LLM: {args.model_name}")
    try:
        llm = AzureOpenAIChatWrapper(
            model_name=args.model_name,
            max_completion_tokens=args.max_new_tokens,
            reasoning_effort=args.reasoning_effort,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Generate hints with progress bar
    print(f"Generating hints with {args.max_concurrent} concurrent requests...")

    tasks = [
        process_sample(llm, sample, args.max_retries, args.retry_delay, semaphore)
        for sample in samples
    ]

    results = []
    for idx, coro in enumerate(
        async_tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Generating hints"
        )
    ):
        result = await coro
        results.append(result)

        # Checkpoint saving
        if args.checkpoint_interval > 0 and (idx + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"hints_checkpoint_{idx + 1}.jsonl"
            )
            save_results(results, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Sort results by original ID to maintain order
    results.sort(key=lambda x: x["id"])

    # Save final results
    output_path = os.path.join(args.output_dir, "hints.jsonl")
    save_results(results, output_path)

    # Print statistics
    successful = sum(1 for r in results if r.get("generation_success", False))
    failed = len(results) - successful

    print(f"\n{'=' * 50}")
    print(f"Generation complete!")
    print(f"Total samples: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful / len(results) * 100:.1f}%")
    print(f"Output saved to: {output_path}")
    print(f"{'=' * 50}")


def main(args):
    """Wrapper to run async main function."""
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args)
