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

def run_benchmark():
    # --- Configuration ---
    base_dataset_name = "train"  # Or your target dataset like "beauty", "sports", etc.
    # Make sure the dataset files (e.g., ml-1m.txt) are in the root of the project or accessible.
    
    base_train_dir_prefix = "benchmark"
    results_log_file = "benchmark_results.jsonl" # Store results in a JSONL file

    # Define hyperparameter ranges you want to test
    # Example:
    learning_rates = [0.001, 0.0005]
    max_lengths = [50, 100] # Max sequence length
    num_blocks_list = [1, 2]  # Number of SASRec blocks
    hidden_units_list = [50] # Number of hidden units
    num_epochs_list = [2, 5] # Small number of epochs for quick testing; increase for real benchmarks
    dropout_rates = [0.2, 0.5]
    l2_emb_values = [0.0]
    num_heads_list = [1]
    
    # Device to run on
    device = "cpu" # Change to "cuda" if a GPU is available and desired

    # --- End Configuration ---

    # Get default arguments from main.py's parser
    default_args = main_parser.parse_args(["--dataset", base_dataset_name, "--train_dir", "temp_default"])

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
    
    all_run_results = []

    for i, params_combo in enumerate(all_hyperparameter_combinations):
        lr, maxlen, num_blocks, hidden_units, num_epochs, dropout, l2, num_heads = params_combo

        run_args = argparse.Namespace(**vars(default_args)) # Create a copy of default args

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
        run_args.save_files = True # Important for data partitioning if not already done for this dataset_train_dir

        # Create a descriptive train_dir for this run
        run_specific_train_dir = f"{base_train_dir_prefix}_{base_dataset_name}_lr{lr}_ml{maxlen}_nb{num_blocks}_hu{hidden_units}_ep{num_epochs}_dr{dropout}_l2{l2}_nh{num_heads}_{int(time.time())}"
        run_args.train_dir = run_specific_train_dir
        run_args.dataset = base_dataset_name # Ensure dataset is correctly set for each run

        print(f"\n--- Running Combination {i+1}/{len(all_hyperparameter_combinations)} ---")
        print(f"Parameters: lr={lr}, maxlen={maxlen}, num_blocks={num_blocks}, hidden_units={hidden_units}, epochs={num_epochs}, dropout={dropout}, l2={l2}, heads={num_heads}")
        print(f"Train directory: models/{base_dataset_name}_{run_specific_train_dir}")
        
        start_time = time.time()
        
        current_run_summary = None # Initialize before try block
        try:
            results = main_process(run_args)
            run_duration = time.time() - start_time
            
            current_run_summary = {
                "params": {
                    "lr": lr, "maxlen": maxlen, "num_blocks": num_blocks, 
                    "hidden_units": hidden_units, "num_epochs": num_epochs, 
                    "dropout_rate": dropout, "l2_emb": l2, "num_heads": num_heads,
                    "dataset": base_dataset_name, "train_dir": run_specific_train_dir
                },
                "results": results,
                "duration_seconds": run_duration
            }
            all_run_results.append(current_run_summary)

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
            current_run_summary = {
                "params": {
                    "lr": lr, "maxlen": maxlen, "num_blocks": num_blocks, 
                    "hidden_units": hidden_units, "num_epochs": num_epochs, 
                    "dropout_rate": dropout, "l2_emb": l2, "num_heads": num_heads,
                    "dataset": base_dataset_name, "train_dir": run_specific_train_dir
                },
                "results": {"status": "CRASH", "error_message": str(e)},
                "duration_seconds": run_duration
            }
            all_run_results.append(current_run_summary)
        
        finally:
            if current_run_summary: # Only log if summary was created (either success or caught exception)
                with open(results_log_file, 'a') as f_log:
                    json.dump(current_run_summary, f_log)
                    f_log.write('\n')
                print(f"Results/error for this run appended to {results_log_file}")


    print(f"\n--- Benchmark Complete ---")
    print(f"Total combinations run: {len(all_run_results)}")
    print(f"All results logged to {results_log_file}")

    # You can add more sophisticated analysis here if needed, 
    # for example, finding the best performing combination.
    # best_run = None
    # highest_ndcg = -1
    # for r_data in all_run_results:
    #     if r_data.get('results', {}).get('status') == 'success':
    #         current_ndcg = r_data.get('results', {}).get('metrics', {}).get('best_val_ndcg_at_10', -1)
    #         if current_ndcg > highest_ndcg:
    #             highest_ndcg = current_ndcg
    #             best_run = r_data

    # print("\n--- Best Run (based on best_val_ndcg_at_10) ---")
    # if best_run:
    #     print(f"Parameters: {best_run['params']}")
    #     print(f"Metrics: {best_run['results']['metrics']}")
    # else:
    #     print("Could not determine a best run or no successful runs.")


if __name__ == "__main__":
    # Before running, ensure that:
    # 1. The dataset file (e.g., "ml-1m.txt") exists in the root directory of your project,
    #    or adjust paths in `solution/utils.py` if it expects data elsewhere.
    # 2. You have sufficient disk space for models and logs, especially in the `models/` directory.
    # 3. The required packages (torch, numpy, etc.) are installed.
    
    # Example: To run this script, navigate to the `solution` directory and run:
    # python benchmark_runner.py
    # Or, if your project root is `recommender-systems-project`, and `solution` is a module:
    # python -m solution.benchmark_runner

    run_benchmark() 