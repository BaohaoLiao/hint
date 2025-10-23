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
        default=8000,
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
