import os
import time
import torch
import argparse
from tqdm import tqdm

from model import SASRec
from utils import *


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--maxlen", default=200, type=int)
parser.add_argument("--hidden_units", default=50, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=1000, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument("--l2_emb", default=0.0, type=float)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--inference_only", default=False, type=str2bool)
parser.add_argument("--state_dict_path", default=None, type=str)
parser.add_argument("--generate_recommendations", default=False, type=str2bool)
parser.add_argument("--top_n", default=10, type=int)
parser.add_argument(
    "--save_files",
    default=True,
    type=str2bool,
    help="Whether to save train/validation/test splits to files",
)
parser.add_argument(
    "--training_dataset",
    default=None,
    type=str,
    help="Dataset used for training (for usernum/itemnum in inference/recommendation)",
)
parser.add_argument(
    "--explicit_negatives",
    default=False,
    type=str2bool,
    help="If True, sample negatives from both disliked and unseen items. If False, only sample from unseen items.",
)
parser.add_argument(
    "--weighted_dislike",
    default=False,
    type=str2bool,
    help="If True, upweight the loss for disliked negatives during training.",
)

# args = parser.parse_args() # Moved to if __name__ == "__main__"

# Place all experiment folders inside 'models'
# models_root = "models" # Moved to main_process
# dataset_train_dir = os.path.join(models_root, args.dataset + "_" + args.train_dir) # Moved to main_process
# os.makedirs(dataset_train_dir, exist_ok=True) # Moved to main_process

# with open(os.path.join(dataset_train_dir, "args.txt"), "w") as f: # Moved to main_process
#     f.write(
#         "\\n".join(
#             [
#                 str(k) + "," + str(v)
#                 for k, v in sorted(vars(args).items(), key=lambda x: x[0])
#             ]
#         )
#     )
# f.close() # Moved to main_process


def main_process(args):
    models_root = "models"
    # Add suffixes to train_dir based on flags
    train_dir_name = args.train_dir
    if args.explicit_negatives and args.weighted_dislike:
        train_dir_name += "_explicit_negatives_with_weighted_dislike"
    elif args.explicit_negatives:
        train_dir_name += "_explicit_negatives"
    dataset_train_dir = os.path.join(models_root, args.dataset + "_" + train_dir_name)
    os.makedirs(dataset_train_dir, exist_ok=True)

    with open(os.path.join(dataset_train_dir, "args.txt"), "w") as f_args:
        f_args.write(
            "\\n".join(
                [
                    str(k) + "," + str(v)
                    for k, v in sorted(vars(args).items(), key=lambda x: x[0])
                ]
            )
        )
    # f_args.close() # With 'with open', close is automatic

    # Decide if we want all interactions in the test set (cross-dataset inference/recommendation)
    all_in_test = False
    if (
        args.inference_only or args.generate_recommendations
    ) and args.training_dataset is not None:
        all_in_test = True

    # If cross-dataset inference/recommendation, build splits accordingly
    if (args.inference_only or args.generate_recommendations) and args.training_dataset is not None:
        # Load training dataset splits (for user histories)
        train_dataset = data_partition(
            args.training_dataset,
            save_files=False,
            out_dir=None,
            all_in_test=False,
        )
        # Load evaluation dataset splits (for test items)
        eval_dataset = data_partition(
            args.dataset,
            save_files=args.save_files,
            out_dir=dataset_train_dir,
            all_in_test=True,
        )
        user_train, user_valid, user_test, usernum, itemnum = build_cross_dataset_splits(train_dataset, eval_dataset)
        # Optionally, also merge UserLiked/UserDisliked if needed
        UserLiked, UserDisliked = train_dataset[0], train_dataset[1]
    else:
        # Standard single-dataset split
        UserLiked, UserDisliked, user_train, user_valid, user_test, usernum, itemnum = data_partition(
            args.dataset,
            save_files=args.save_files,
            out_dir=dataset_train_dir,
            all_in_test=False,
        )

    # Override usernum and itemnum if a different training dataset's stats are specified for model compatibility
    if args.training_dataset is not None and (
        args.inference_only or args.generate_recommendations
    ):
        print(
            f"Using user/item counts from specified training dataset: {args.training_dataset}"
        )
        usernum_from_training_set, itemnum_from_training_set = get_user_item_counts(
            args.training_dataset
        )
        usernum = usernum_from_training_set
        itemnum = itemnum_from_training_set
        print(
            f"Overridden usernum: {usernum}, itemnum: {itemnum} for model initialization."
        )

    model = SASRec(usernum, itemnum, args).to(args.device)

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(
                torch.load(args.state_dict_path, map_location=torch.device(args.device))
            )
            print(f"Loaded model weights from {args.state_dict_path}")
        except Exception as e:
            error_msg = (
                f"Failed loading state_dicts from {args.state_dict_path}. Error: {e}"
            )
            print(error_msg)
            if args.generate_recommendations:
                return {
                    "status": "failure",
                    "mode": "generate_recommendations",
                    "error": f"{error_msg} Cannot generate recommendations without a loaded model.",
                }
            elif args.inference_only:
                return {
                    "status": "failure",
                    "mode": "inference",
                    "error": f"{error_msg} Cannot run inference without model weights.",
                }
            else:  # Training mode
                print(
                    "Proceeding with training from scratch or without resumed state due to load failure."
                )
                # Model remains in its initial state, epoch_start_idx logic will handle starting from epoch 1.
    elif (
        args.generate_recommendations
    ):  # generate_recommendations requires state_dict_path
        error_msg = "Error: --generate_recommendations mode requires --state_dict_path to be provided."
        print(error_msg)
        return {
            "status": "failure",
            "mode": "generate_recommendations",
            "error": error_msg,
        }

    if args.generate_recommendations:
        print(f"Generating top-{args.top_n} recommendations for each user...")
        model.eval()

        all_item_ids = list(range(1, itemnum + 1))
        recommendations = {}

        progress_bar = tqdm(
            range(1, usernum + 1),
            desc="Generating recommendations",
            ncols=100,
            position=0,
            leave=True,
        )

        for user_id in progress_bar:
            # Update progress description with current user
            progress_bar.set_description(f"Processing user {user_id}/{usernum}")

            # Ensure user_train, user_valid, user_test are available. They should be from dataset_splits.
            history_train = user_train.get(user_id, [])
            history_valid = user_valid.get(user_id, [])
            history_test = user_test.get(user_id, [])

            user_full_history = history_train + history_valid + history_test
            user_full_history_set = set(user_full_history)

            if not user_full_history:
                progress_bar.write(f"User {user_id} has no history. Skipping.")
                recommendations[user_id] = []
                continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(user_full_history):
                if idx == -1:
                    break
                seq[idx] = i
                idx -= 1

            items_to_score = [
                item_id
                for item_id in all_item_ids
                if item_id not in user_full_history_set
            ]

            if not items_to_score:
                progress_bar.write(
                    f"User {user_id} has interacted with all items. Skipping."
                )
                recommendations[user_id] = []
                continue

            log_seqs_np = np.array([seq])
            item_indices_np = np.array(items_to_score)

            scores = model.predict(np.array([user_id]), log_seqs_np, item_indices_np)
            scores_np = scores.cpu().detach().numpy().flatten()

            scored_items = list(zip(scores_np, items_to_score))
            scored_items.sort(key=lambda x: x[0], reverse=True)

            top_n_recommendations = [
                item_id for score, item_id in scored_items[: args.top_n]
            ]
            recommendations[user_id] = top_n_recommendations

        progress_bar.close()
        print("\n--- Top-N Recommendations Generated ---")

        recs_file_path = os.path.join(dataset_train_dir, "recommendations.json")
        # Save recommendations in the same folder as the model weights used
        if args.state_dict_path is not None:
            model_folder = os.path.dirname(args.state_dict_path)
        else:
            model_folder = dataset_train_dir  # fallback

        recs_file_path = os.path.join(model_folder, f"top_{args.top_n}.json")
        try:
            import json

            with open(recs_file_path, "w") as f_recs:
                json.dump(recommendations, f_recs)
            print(f"Recommendations saved to {recs_file_path}")
            return {
                "status": "success",
                "mode": "recommendation_generation",
                "output_file": recs_file_path,
                "num_users_processed": len(recommendations),
            }
        except Exception as e:
            error_msg = f"Failed to save recommendations: {e}"
            print(error_msg)
            return {
                "status": "failure",
                "mode": "recommendation_generation",
                "error": error_msg,
                "recommendations_data": recommendations,
            }

    elif args.inference_only:
        model.eval()
        t_test = evaluate(
            model, (user_train, user_valid, user_test, usernum, itemnum), args
        )  # Pass dataset_splits
        print(
            "test (NDCG@10: %.4f, P@10: %.4f, R@10: %.4f)"
            % (t_test[0], t_test[1], t_test[2])
        )
        num_valid_items = sum(len(v) for v in user_valid.values())
        num_test_items = sum(len(v) for v in user_test.values())
        num_valid_users = len(user_valid)
        num_test_users = len(user_test)
        avg_valid_per_user = num_valid_items / num_valid_users if num_valid_users > 0 else 0
        avg_test_per_user = num_test_items / num_test_users if num_test_users > 0 else 0

        return {
            "status": "success",
            "mode": "inference",
            "metrics": {
                "ndcg_at_10": t_test[0],
                "p_at_10": t_test[1],
                "r_at_10": t_test[2],
                "num_validation_items": num_valid_items,
                "num_test_items": num_test_items,
                "avg_validation_items_per_user": avg_valid_per_user,
                "avg_test_items_per_user": avg_test_per_user,
            },
        }

    else:  # Training mode
        num_batch = (
            (len(user_train) - 1) // args.batch_size + 1 if len(user_train) > 0 else 0
        )
        if num_batch == 0:
            print(
                "Warning: No training data found or user_train is empty. Cannot train."
            )
            return {
                "status": "failure",
                "mode": "training",
                "error": "No training data available.",
            }

        total_interactions = 0.0
        for u_id in user_train:
            total_interactions += len(user_train[u_id])
        print(
            "average sequence length: %.2f"
            % (total_interactions / len(user_train) if len(user_train) > 0 else 0.0)
        )

        log_file_path = os.path.join(dataset_train_dir, "log.txt")
        with open(log_file_path, "w") as f_log:
            f_log.write(
                "epoch (val_ndcg@10, val_p@10, val_r@10) (test_ndcg@10, test_p@10, test_r@10)\\n"
            )

            sampler = WarpSampler(
                user_train,
                usernum,
                itemnum,
                batch_size=args.batch_size,
                maxlen=args.maxlen,
                n_workers=3,
                explicit_negatives=args.explicit_negatives,
                user_disliked=UserDisliked,
            )

            for name, param in model.named_parameters():
                try:
                    torch.nn.init.xavier_normal_(param.data)
                except:
                    pass

            model.pos_emb.weight.data[0, :] = 0
            model.item_emb.weight.data[0, :] = 0

            model.train()

            epoch_start_idx = 1
            if (
                args.state_dict_path is not None and not args.generate_recommendations
            ):  # Check if resuming and model was actually loaded earlier
                # Model loading attempt was done earlier. If it failed, training starts from scratch.
                # If it succeeded, try to parse epoch.
                if any(
                    p.requires_grad for p in model.parameters()
                ):  # A proxy to check if model was loaded vs fresh
                    try:
                        # Check if model actually has loaded weights, e.g. by comparing to a fresh init
                        # For simplicity, just try to parse epoch if path was given.
                        # The earlier check for args.state_dict_path already covers if the path was provided.
                        # And the load attempt was made.
                        # If model.load_state_dict was successful:
                        if os.path.exists(
                            args.state_dict_path
                        ):  # Re-check to be sure, as load_state_dict might not error if path is wrong but model init is same
                            # This logic assumes model was loaded. The earlier exception handles load failure.
                            tail = args.state_dict_path[
                                args.state_dict_path.find("epoch=") + 6 :
                            ]
                            epoch_start_idx = int(tail[: tail.find(".")]) + 1
                            print(f"Resuming training from epoch {epoch_start_idx}")
                    except:
                        print(
                            "Failed to parse epoch from state_dict_path for training resumption. Starting from epoch 1."
                        )
                        epoch_start_idx = 1

            bce_criterion = torch.nn.BCEWithLogitsLoss()
            adam_optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, betas=(0.9, 0.98)
            )

            best_val_ndcg, best_val_precision, best_val_recall = 0.0, 0.0, 0.0
            best_test_ndcg, best_test_precision, best_test_recall = 0.0, 0.0, 0.0
            total_training_time = 0.0
            t0 = time.time()
            epoch_iterator = tqdm(
                range(epoch_start_idx, args.num_epochs + 1),
                desc="Training Progress",
                position=0,
                leave=True,
                ncols=100,
            )

            for epoch in epoch_iterator:
                batch_iterator = tqdm(
                    range(num_batch),
                    desc=f"Epoch {epoch}/{args.num_epochs}",
                    position=1,
                    leave=False,
                    ncols=100,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                )
                for step in batch_iterator:
                    u, seq, pos, neg, neg_weight = sampler.next_batch()
                    u, seq, pos, neg, neg_weight = (
                        np.array(u),
                        np.array(seq),
                        np.array(pos),
                        np.array(neg),
                        np.array(neg_weight),
                    )
                    neg_weight = torch.tensor(
                        neg_weight, dtype=torch.float32, device=args.device
                    )

                    pos_logits, neg_logits = model(u, seq, pos, neg)
                    pos_labels, neg_labels = torch.ones(
                        pos_logits.shape, device=args.device
                    ), torch.zeros(neg_logits.shape, device=args.device)

                    adam_optimizer.zero_grad()
                    indices = np.where(pos != 0)
                    loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    if args.weighted_dislike:
                        # Use weighted loss for negatives
                        loss += torch.nn.BCEWithLogitsLoss(weight=neg_weight[indices])(
                            neg_logits[indices], neg_labels[indices]
                        )
                    else:
                        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                    for param in model.item_emb.parameters():
                        loss += args.l2_emb * torch.norm(param)
                    loss.backward()
                    adam_optimizer.step()

                    batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

                # Clear the batch progress bar at epoch end
                batch_iterator.close()

                if (
                    epoch % 20 == 0 or epoch == args.num_epochs
                ):  # Evaluate every 20 epochs or at the last epoch
                    model.eval()
                    t1 = time.time() - t0
                    total_training_time += t1

                    # Temporarily clear progress bars for evaluation output
                    epoch_iterator.clear()

                    t_test_eval = evaluate(
                        model,
                        (user_train, user_valid, user_test, usernum, itemnum),
                        args,
                    )
                    t_valid_eval = evaluate(
                        model,
                        (user_train, user_valid, user_test, usernum, itemnum),
                        args,
                        mode="valid",
                    )

                    eval_msg = (
                        f"epoch:{epoch}, time: {total_training_time:.2f}(s), "
                        f"valid (NDCG@10: {t_valid_eval[0]:.4f}, P@10: {t_valid_eval[1]:.4f}, R@10: {t_valid_eval[2]:.4f}), "
                        f"test (NDCG@10: {t_test_eval[0]:.4f}, P@10: {t_test_eval[1]:.4f}, R@10: {t_test_eval[2]:.4f})"
                    )

                    # Update progress bar description with evaluation results
                    epoch_iterator.write(eval_msg)

                    # Write to log file
                    f_log.write(
                        str(epoch)
                        + " "
                        + str(t_valid_eval)
                        + " "
                        + str(t_test_eval)
                        + "\n"
                    )
                    f_log.flush()

                    if (
                        t_valid_eval[2] > best_val_recall
                        or t_valid_eval[0] > best_val_ndcg
                    ):
                        epoch_iterator.write(
                            f"New best validation performance found at epoch {epoch}."
                        )
                        best_val_ndcg = t_valid_eval[0]
                        best_val_precision = t_valid_eval[1]
                        best_val_recall = t_valid_eval[2]

                        best_test_ndcg = t_test_eval[0]
                        best_test_precision = t_test_eval[1]
                        best_test_recall = t_test_eval[2]

                        folder = dataset_train_dir
                        fname = "SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth"
                        fname = fname.format(
                            epoch,
                            args.lr,
                            args.num_blocks,
                            args.num_heads,
                            args.hidden_units,
                            args.maxlen,
                        )
                        torch.save(model.state_dict(), os.path.join(folder, fname))
                        epoch_iterator.write(
                            f"Saved model to {os.path.join(folder, fname)}"
                        )

                    t0 = time.time()
                    model.train()

            # Clean up at the end
            epoch_iterator.close()
        num_valid_items = sum(len(v) for v in user_valid.values())
        num_test_items = sum(len(v) for v in user_test.values())
        num_valid_users = len(user_valid)
        num_test_users = len(user_test)
        avg_valid_per_user = num_valid_items / num_valid_users if num_valid_users > 0 else 0
        avg_test_per_user = num_test_items / num_test_users if num_test_users > 0 else 0

        return {
            "status": "success",
            "mode": "training",
            "metrics": {
                "best_val_ndcg_at_10": best_val_ndcg,
                "best_val_p_at_10": best_val_precision,
                "best_val_r_at_10": best_val_recall,
                "corresponding_test_ndcg_at_10": best_test_ndcg,
                "corresponding_test_p_at_10": best_test_precision,
                "corresponding_test_r_at_10": best_test_recall,
                "log_file": log_file_path,
                "num_validation_items": num_valid_items,
                "num_test_items": num_test_items,
                "avg_validation_items_per_user": avg_valid_per_user,
                "avg_test_items_per_user": avg_test_per_user,
            },
        }


if __name__ == "__main__":
    args = parser.parse_args()

    if args.weighted_dislike:
        args.explicit_negatives = True

    results = main_process(args)

    print("\n--- Process Summary ---")
    if results:
        print(f"Status: {results.get('status')}")
        print(f"Mode: {results.get('mode')}")
        if "metrics" in results:
            print("Metrics:")
            for k, v in results["metrics"].items():
                print(f"  {k}: {v}")
        if "error" in results:
            print(f"Error: {results.get('error')}")
        if "output_file" in results:
            print(f"Output File: {results.get('output_file')}")
    else:
        print("main_process did not return any results.")

    print("\nDone")
