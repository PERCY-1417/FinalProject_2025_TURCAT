import argparse
import itertools
import os
import time
import json

# Assuming 'solution' is in PYTHONPATH or benchmark_runner.py is in the same directory as main.py
# If not, you might need to adjust sys.path
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Or adjust as needed

from main import main_process, parser as main_parser

def load_and_process_results(results_file_path):
    """Loads results from a JSONL file, filters for successful runs, and extracts data."""
    processed_runs = []
    if not os.path.exists(results_file_path):
        print(f"Results file not found: {results_file_path}")
        return processed_runs

    with open(results_file_path, 'r') as f_in:
        for line in f_in:
            try:
                log_entry = json.loads(line.strip())
                if (
                    log_entry.get("results")
                    and log_entry["results"].get("status") == "success"
                    and "metrics" in log_entry["results"]
                    and log_entry["results"].get("mode") == "training" # Focus on training results for benchmark tables
                ):
                    params = log_entry.get("params", {})
                    metrics = log_entry["results"].get("metrics", {})
                    duration = log_entry.get("duration_seconds", 0)
                    
                    processed_runs.append({
                        "lr": params.get("lr"),
                        "maxlen": params.get("maxlen"),
                        "num_blocks": params.get("num_blocks"),
                        "hidden_units": params.get("hidden_units"),
                        "num_epochs": params.get("num_epochs"),
                        "dropout_rate": params.get("dropout_rate"),
                        "best_val_ndcg_at_10": metrics.get("best_val_ndcg_at_10"),
                        "corresponding_test_ndcg_at_10": metrics.get("corresponding_test_ndcg_at_10"),
                        "best_val_r_at_10": metrics.get("best_val_r_at_10"),
                        "corresponding_test_r_at_10": metrics.get("corresponding_test_r_at_10"),
                        "duration_seconds": duration,
                        # Store original params and full metrics if needed for deeper analysis later
                        # "original_params": params,
                        # "original_metrics": metrics 
                    })
            except json.JSONDecodeError:
                print(f"Skipping malformed line in {results_file_path}: {line.strip()}")
            except Exception as e:
                print(f"Error processing line: {line.strip()} - {e}")
    return processed_runs

def display_results_table(processed_data, top_n=10, sort_metric='best_val_ndcg_at_10'):
    """Sorts processed data and prints a formatted table of top N results."""
    if not processed_data:
        print("No successful training runs found to display in table.")
        return

    # Sort data
    # Handle cases where the sort_metric might be None (e.g., if a run failed to produce it)
    sorted_data = sorted(
        processed_data,
        key=lambda x: x.get(sort_metric, -1) if x.get(sort_metric) is not None else -1, # Use -1 for None or missing
        reverse=True
    )

    top_results = sorted_data[:top_n]

    print(f"\n--- Top {len(top_results)} Benchmark Results (Sorted by {sort_metric}) ---")
    header = (
        "Rank | LR      | Maxlen | Blocks | Hidden | Epochs | Dropout | Val NDCG | Test NDCG | Val R@10 | Test R@10 | Duration (s)"
    )
    print(header)
    print("-" * len(header))

    for i, run in enumerate(top_results):
        print(
            f"{i+1:<4} | "
            f"{run.get('lr', 'N/A'):<7.5f} | "
            f"{run.get('maxlen', 'N/A'):<6} | "
            f"{run.get('num_blocks', 'N/A'):<6} | "
            f"{run.get('hidden_units', 'N/A'):<6} | "
            f"{run.get('num_epochs', 'N/A'):<6} | "
            f"{run.get('dropout_rate', 'N/A'):<7.3f} | "
            f"{run.get('best_val_ndcg_at_10', 0.0):<8.4f} | "
            f"{run.get('corresponding_test_ndcg_at_10', 0.0):<9.4f} | "
            f"{run.get('best_val_r_at_10', 0.0):<8.4f} | " 
            f"{run.get('corresponding_test_r_at_10', 0.0):<9.4f} | " 
            f"{run.get('duration_seconds', 0.0):<12.2f}"
        )
    print("-" * len(header))


def run_benchmark(args_cli):
    # --- Configuration --- (Mostly from args_cli or defaults now)
    base_dataset_name = args_cli.benchmark_dataset_name
    base_train_dir_prefix = "benchmark"
    results_log_file = args_cli.results_log_file

    # Define hyperparameter ranges you want to test
    learning_rates = args_cli.learning_rates
    max_lengths = args_cli.max_lengths
    num_blocks_list = args_cli.num_blocks
    hidden_units_list = args_cli.hidden_units
    num_epochs_list = args_cli.num_epochs
    dropout_rates = args_cli.dropout_rates
    l2_emb_values = args_cli.l2_emb_values
    num_heads_list = args_cli.num_heads
    
    device = args_cli.device

    # --- End Configuration ---

    # Get default arguments from main.py's parser
    # We need to pass at least dataset and train_dir for main_parser to not fail
    default_main_args = main_parser.parse_args(["--dataset", base_dataset_name, "--train_dir", "temp_default_for_init"])

    all_hyperparameter_combinations = list(itertools.product(
        learning_rates,
        max_lengths,
        num_blocks_list,
        hidden_units_list,
        num_epochs_list,
        dropout_rates,
        l2_emb_values,
        num_heads_list
    ))

    print(f"Starting benchmark with {len(all_hyperparameter_combinations)} combinations.")
    print(f"Results will be logged to: {results_log_file}")
    if os.path.exists(results_log_file):
        print(f"Note: {results_log_file} already exists. New results will be appended.")

    all_run_results_summary_objects = [] # Stores summary objects for table display later

    for i, params_combo in enumerate(all_hyperparameter_combinations):
        lr, maxlen, num_blocks, hidden_units, num_epochs, dropout, l2, num_heads = params_combo

        run_args = argparse.Namespace(**vars(default_main_args)) # Create a copy of default args

        # Override with current combination
        run_args.lr = lr
        run_args.maxlen = maxlen
        run_args.num_blocks = num_blocks
        run_args.hidden_units = hidden_units
        run_args.num_epochs = num_epochs
        run_args.dropout_rate = dropout
        run_args.l2_emb = l2
        run_args.num_heads = num_heads
        
        run_args.device = device
        run_args.inference_only = False # We want to train and evaluate
        run_args.generate_recommendations = False
        run_args.state_dict_path = None # Train from scratch for each benchmark run
        run_args.save_files = True

        run_specific_train_dir = f"{base_train_dir_prefix}_{base_dataset_name}_lr{lr}_ml{maxlen}_nb{num_blocks}_hu{hidden_units}_ep{num_epochs}_dr{dropout}_l2{l2}_nh{num_heads}_{int(time.time())}"
        run_args.train_dir = run_specific_train_dir
        run_args.dataset = base_dataset_name

        print(f"\n--- Running Combination {i+1}/{len(all_hyperparameter_combinations)} ---")
        print(f"Parameters: lr={lr}, maxlen={maxlen}, num_blocks={num_blocks}, hidden_units={hidden_units}, epochs={num_epochs}, dropout={dropout}, l2={l2}, heads={num_heads}")
        print(f"Train directory for this run: models/{base_dataset_name}_{run_specific_train_dir}") # Adjusted path to show actual structure
        
        start_time = time.time()
        current_run_log_entry = None
        try:
            results = main_process(run_args)
            run_duration = time.time() - start_time
            
            current_run_log_entry = {
                "params": {
                    "lr": lr, "maxlen": maxlen, "num_blocks": num_blocks, 
                    "hidden_units": hidden_units, "num_epochs": num_epochs, 
                    "dropout_rate": dropout, "l2_emb": l2, "num_heads": num_heads,
                    "dataset": base_dataset_name, "train_dir": run_specific_train_dir
                },
                "results": results, # This is the dict returned by main_process
                "duration_seconds": run_duration
            }
            all_run_results_summary_objects.append(current_run_log_entry) # For final table display

            print(f"Finished in {run_duration:.2f} seconds.")
            if results:
                print(f"Status: {results.get('status')}")
                if "metrics" in results:
                    print("Metrics:")
                    for k, v_metric in results["metrics"].items():
                        print(f"  {k}: {v_metric}")
                if "error" in results:
                    print(f"Error: {results.get('error')}")
            else:
                print("main_process did not return any results for this combination.")

        except Exception as e:
            run_duration = time.time() - start_time
            print(f"An error occurred during main_process for this combination: {e}")
            current_run_log_entry = {
                "params": {
                    "lr": lr, "maxlen": maxlen, "num_blocks": num_blocks, 
                    "hidden_units": hidden_units, "num_epochs": num_epochs, 
                    "dropout_rate": dropout, "l2_emb": l2, "num_heads": num_heads,
                    "dataset": base_dataset_name, "train_dir": run_specific_train_dir
                },
                "results": {"status": "CRASH", "error_message": str(e), "mode": "training"},
                "duration_seconds": run_duration
            }
            all_run_results_summary_objects.append(current_run_log_entry)
        
        finally:
            if current_run_log_entry:
                with open(results_log_file, 'a') as f_log:
                    json.dump(current_run_log_entry, f_log)
                    f_log.write('\n')
                print(f"Log entry for this run appended to {results_log_file}")

    print(f"\n--- Benchmark Run Complete ---")
    print(f"Total combinations processed: {len(all_hyperparameter_combinations)}")
    print(f"All detailed logs in: {results_log_file}")

    # Display table from the just-completed run
    # We use all_run_results_summary_objects which contains the structured data, 
    # rather than re-reading and parsing the entire file immediately if it's very large.
    # However, for consistency and to show how load_and_process_results works, we can call it.
    # If the file is huge, it might be slightly slower but ensures we're using the persisted data.
    
    # For immediate display from current session (potentially faster if results_log_file is huge)
    # processed_current_session_data = [run for run in all_run_results_summary_objects if run["results"].get("status") == "success" and run["results"].get("mode") == "training"]
    # display_results_table(processed_current_session_data, top_n=args_cli.top_n, sort_metric=args_cli.sort_metric)
    
    # To display from the file (more robust if runs are very long and you only want final analysis)
    print("\nLoading all results from file for final table display...")
    all_processed_data_from_file = load_and_process_results(results_log_file)
    display_results_table(all_processed_data_from_file, top_n=args_cli.top_n, sort_metric=args_cli.sort_metric)

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Benchmark runner for SASRec model.")
    cli_parser.add_argument("--analyze", type=str, metavar="FILEPATH", help="Path to benchmark_results.jsonl file to analyze. Skips running new benchmarks.")
    cli_parser.add_argument("--results_log_file", type=str, default="benchmark_results.jsonl", help="File to log benchmark results.")
    cli_parser.add_argument("--top_n", type=int, default=10, help="Number of top results to display in the summary table.")
    cli_parser.add_argument("--sort_metric", type=str, default='best_val_ndcg_at_10', help="Metric to sort results by (e.g., best_val_ndcg_at_10, corresponding_test_ndcg_at_10)." )

    # Hyperparameters for benchmark runs (if not in analyze mode)
    cli_parser.add_argument("--benchmark_dataset_name", type=str, default="small_matrix", help="Base dataset name for benchmarking (e.g., small_matrix, beauty).")
    cli_parser.add_argument("--learning_rates", type=float, nargs='+', default=[0.0015, 0.0005])
    cli_parser.add_argument("--max_lengths", type=int, nargs='+', default=[150, 300])
    cli_parser.add_argument("--num_blocks", type=int, nargs='+', default=[2, 4])
    cli_parser.add_argument("--hidden_units", type=int, nargs='+', default=[50])
    cli_parser.add_argument("--num_epochs", type=int, nargs='+', default=[17, 20, 25]) # Small for testing
    cli_parser.add_argument("--dropout_rates", type=float, nargs='+', default=[0.5, 0.7])
    cli_parser.add_argument("--l2_emb_values", type=float, nargs='+', default=[0.0])
    cli_parser.add_argument("--num_heads", type=int, nargs='+', default=[1])
    cli_parser.add_argument("--device", type=str, default="cpu", help="Device to run on (e.g., cpu, cuda).")

    args_cli = cli_parser.parse_args()

    if args_cli.analyze:
        print(f"Analyzing results from: {args_cli.analyze}")
        processed_data = load_and_process_results(args_cli.analyze)
        display_results_table(processed_data, top_n=args_cli.top_n, sort_metric=args_cli.sort_metric)
    else:
        # Before running, ensure that:
        # 1. The dataset file (e.g., "small_matrix.txt") exists in the root directory of your project,
        #    or adjust paths in `solution/utils.py` if it expects data elsewhere.
        # 2. You have sufficient disk space for models and logs, especially in the `models/` directory.
        # 3. The required packages (torch, numpy, etc.) are installed.
        
        # Example: To run this script, navigate to the `solution` directory and run:
        # python benchmark_runner.py --learning_rates 0.001 --num_epochs 10 --max_lengths 50
        # To analyze existing results:
        # python benchmark_runner.py --analyze benchmark_results.jsonl --top_n 5

        run_benchmark(args_cli) 