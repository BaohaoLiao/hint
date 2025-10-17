#######################################################################
# Merge results from all GPUs across all iterations
#######################################################################

import os
import chz
import json
import sys
from collections import defaultdict


@chz.chz
class CLIConfig:
    # iteration
    current_iteration: int = 1  # Current iteration number that just completed
    world_size: int = 8  # Number of GPUs used

    # Save
    output_dir: str = "gen_dataset"
    merged_file: str = "merged_all_iterations.json"


def load_previous_merged_results(merged_file_path):
    """Load previous merged results if they exist."""
    if not os.path.exists(merged_file_path):
        return {}
    
    results = {}
    with open(merged_file_path, 'r', encoding='utf8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                results[sample["problem"]] = sample
    
    return results


def load_current_iteration_results(output_dir, iteration, world_size):
    """Load results from all GPUs for the current iteration."""
    current_results = defaultdict(lambda: {
        "responses": [],
        "scores": []
    })
    
    for gpu_idx in range(world_size):
        iter_file = os.path.join(output_dir, f"iter{iteration}_{gpu_idx}.json")
        
        if not os.path.exists(iter_file):
            print(f"ERROR: File not found: {iter_file}")
            print(f"Stopping merge process. Not all GPUs completed successfully.")
            sys.exit(2)  # Exit code 2 indicates missing file error
        
        print(f"Loading {iter_file}...")
        with open(iter_file, 'r', encoding='utf8') as f:
            count = 0
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    problem = sample["problem"]
                    
                    if problem not in current_results:
                        current_results[problem] = {
                            "problem": problem,
                            "answer": sample["answer"],
                            "responses": [],
                            "scores": [],
                            "iteration": iteration
                        }
                    
                    # Append new responses and scores
                    current_results[problem]["responses"].extend(sample["responses"])
                    current_results[problem]["scores"].extend(sample["scores"])
                    count += 1
            
            print(f"  → Loaded {count} problems from GPU {gpu_idx}")
    
    return dict(current_results)


def merge_with_previous(previous_results, current_results):
    """Merge current iteration results with previous iterations."""
    merged = {}
    
    # Start with all previous results
    for problem, data in previous_results.items():
        merged[problem] = data.copy()
    
    # Add or update with current iteration results
    for problem, current_data in current_results.items():
        if problem in merged:
            # Problem exists from previous iterations - append new responses
            merged[problem]["responses"].extend(current_data["responses"])
            merged[problem]["scores"].extend(current_data["scores"])
            merged[problem]["iteration"] = current_data["iteration"]
        else:
            # New problem (shouldn't happen in normal flow, but handle it)
            merged[problem] = current_data
    
    # Update pass@k=0 flag for all problems
    for problem in merged:
        merged[problem]["pass@k=0"] = not any(merged[problem]["scores"])
    
    return merged


def main(cli_config):
    merged_file_path = os.path.join(cli_config.output_dir, cli_config.merged_file)
    
    print(f"{'='*60}")
    print(f"Merging iteration {cli_config.current_iteration}")
    print(f"{'='*60}")
    
    # Load previous merged results
    print(f"Loading previous merged results from {merged_file_path}...")
    previous_results = load_previous_merged_results(merged_file_path)
    print(f"Found {len(previous_results)} problems from previous iterations")
    
    # Load current iteration results from all GPUs
    print(f"\nLoading current iteration results from {cli_config.world_size} GPUs...")
    current_results = load_current_iteration_results(
        cli_config.output_dir,
        cli_config.current_iteration,
        cli_config.world_size
    )
    print(f"Found {len(current_results)} problems in current iteration")
    
    # Merge results
    print("\nMerging results...")
    merged_results = merge_with_previous(previous_results, current_results)
    
    # Save merged results
    print(f"Saving merged results to {merged_file_path}...")
    with open(merged_file_path, "w", encoding="utf8") as f:
        for sample in merged_results.values():
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")
    
    # Print statistics
    total = len(merged_results)
    passed = sum(1 for s in merged_results.values() if not s["pass@k=0"])
    failed = total - passed
    
    # Calculate average responses per problem
    total_responses = sum(len(s["responses"]) for s in merged_results.values())
    avg_responses = total_responses / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Iteration {cli_config.current_iteration} merge complete!")
    print(f"{'='*60}")
    print(f"Total problems: {total}")
    print(f"Passed: {passed} ({100*passed/total:.2f}%)")
    print(f"Failed: {failed} ({100*failed/total:.2f}%)")
    print(f"Average responses per problem: {avg_responses:.1f}")
    print(f"Total responses collected: {total_responses}")
    print(f"Results saved to: {merged_file_path}")
    print(f"{'='*60}\n")
    
    # Return the number of failed problems for the shell script
    return failed


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    failed_count = main(cli_config)
    
    # Exit with special code if all problems passed
    if failed_count == 0:
        print("✓ All problems have passed!")
        exit(0)
    else:
        print(f"→ {failed_count} problems still need correct solutions")
        exit(1)