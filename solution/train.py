import os
import time
import torch
import argparse
import numpy as np

from model import SASRec
from dataloader import (
    build_index,
    data_partition,
    WarpSampler,
    evaluate,
    evaluate_valid,
)


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


def setup_args():
    """Set up and parse command line arguments."""
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
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--inference_only", default=False, type=str2bool)
    parser.add_argument("--state_dict_path", default=None, type=str)
    return parser.parse_args()


def create_directory_structure(args):
    """Create necessary directories for saving results and logs."""
    output_dir = os.path.abspath(args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([f"{k},{v}" for k, v in sorted(vars(args).items())]))
    return output_dir


def initialize_model(args, usernum, itemnum):
    """Initialize the SASRec model."""
    model = SASRec(usernum, itemnum, args).to(args.device)
    # Initialize model parameters
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    return model


def main():
    args = setup_args()
    output_dir = create_directory_structure(args)

    # Load data and prepare the user-item index
    u2i_index, i2u_index = build_index(args.dataset)
    dataset = data_partition(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum = dataset

    # Calculate batch size and sequence length information
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    avg_seq_len = sum(len(user_train[u]) for u in user_train) / len(user_train)
    print(f"Average sequence length: {avg_seq_len:.2f}")

    # Initialize data sampler
    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=3,
    )

    # Initialize model
    model = initialize_model(args, usernum, itemnum)

    # Load model state if provided
    epoch_start_idx = 1
    if args.state_dict_path:
        try:
            model.load_state_dict(
                torch.load(args.state_dict_path, map_location=torch.device(args.device))
            )
            epoch_start_idx = (
                int(args.state_dict_path.split("epoch=")[1].split(".")[0]) + 1
            )
        except Exception:
            print(f"Failed to load state_dict from {args.state_dict_path}")
            import pdb

            pdb.set_trace()  # Debugging point if needed

    # Optimizer and loss function
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # Log file
    with open(os.path.join(output_dir, "log.txt"), "w") as log_file:
        log_file.write("epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n")

        # Training loop
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            model.train()
            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                u, seq, pos, neg = (
                    np.array(u),
                    np.array(seq),
                    np.array(pos),
                    np.array(neg),
                )

                # Forward pass
                pos_logits, neg_logits = model(u, seq, pos, neg)
                pos_labels = torch.ones(pos_logits.shape, device=args.device)
                neg_labels = torch.zeros(neg_logits.shape, device=args.device)

                # Backward pass
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(
                    pos_logits[indices], pos_labels[indices]
                ) + bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()

                # Logging loss
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

            # Evaluate every 20 epochs
            if epoch % 20 == 0:
                model.eval()
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
                print(
                    f"Epoch {epoch}, Validation NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}"
                )
                print(
                    f"Epoch {epoch}, Test NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f}"
                )

                # Save the best model
                if (
                    t_valid[0] > best_val_ndcg
                    or t_valid[1] > best_val_hr
                    or t_test[0] > best_test_ndcg
                    or t_test[1] > best_test_hr
                ):
                    best_val_ndcg = max(t_valid[0], best_val_ndcg)
                    best_val_hr = max(t_valid[1], best_val_hr)
                    best_test_ndcg = max(t_test[0], best_test_ndcg)
                    best_test_hr = max(t_test[1], best_test_hr)

                    model_path = os.path.join(output_dir, f"SASRec.epoch={epoch}.pth")
                    torch.save(model.state_dict(), model_path)

                # Log the results
                log_file.write(f"{epoch} {t_valid} {t_test}\n")
                log_file.flush()

        # Save the final model
        final_model_path = os.path.join(
            output_dir, f"SASRec.final.epoch={args.num_epochs}.pth"
        )
        torch.save(model.state_dict(), final_model_path)

    sampler.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
