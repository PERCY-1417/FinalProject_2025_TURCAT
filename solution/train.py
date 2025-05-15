import os
import time
import torch
import argparse

from model import SASRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--generate_recommendations', default=False, type=str2bool)
parser.add_argument('--top_n', default=10, type=int)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # Load dataset and model arguments
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    model = SASRec(usernum, itemnum, args).to(args.device)

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            print(f"Loaded model weights from {args.state_dict_path}")
        except Exception as e:
            print(f'Failed loading state_dicts from {args.state_dict_path}. Error: {e}')
            if not args.generate_recommendations: # Allow proceeding without weights if not generating recs (i.e. for training)
                import pdb; pdb.set_trace()
            else:
                print("Cannot generate recommendations without a loaded model. Exiting.")
                exit()
    elif args.generate_recommendations:
        print("Error: --generate_recommendations mode requires --state_dict_path to be provided. Exiting.")
        exit()

    if args.generate_recommendations:
        print(f"Generating top-{args.top_n} recommendations for each user...")
        model.eval()
        
        all_item_ids = list(range(1, itemnum + 1))
        recommendations = {}

        for user_id in range(1, usernum + 1):
            if user_id % 100 == 0:
                print(f"Processing user {user_id}/{usernum}...")

            history_train = user_train.get(user_id, [])
            history_valid = user_valid.get(user_id, [])
            history_test = user_test.get(user_id, [])
            
            # Combine all interactions for the user
            user_full_history = history_train + history_valid + history_test
            user_full_history_set = set(user_full_history)

            if not user_full_history: # Or based on user_train, if we only want to use training sequence
                print(f"User {user_id} has no history. Skipping.")
                recommendations[user_id] = []
                continue

            # Prepare sequence for the model (take last args.maxlen items)
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            # Use the full history to form the sequence fed to the model
            for i in reversed(user_full_history):
                if idx == -1:
                    break
                seq[idx] = i
                idx -= 1
            
            # Determine items to score: all items not in the user's full history
            items_to_score = [item_id for item_id in all_item_ids if item_id not in user_full_history_set]

            if not items_to_score:
                print(f"User {user_id} has interacted with all items. Skipping.")
                recommendations[user_id] = []
                continue
                
            # Get predictions
            # model.predict expects user_ids as (batch_size, 1), log_seqs as (batch_size, maxlen), item_indices as (batch_size, num_items) or (num_items)
            # For single user prediction:
            # user_ids_np = np.array([user_id]).reshape(1, -1) # Shape (1,1) - no, model expects just int user_id
            log_seqs_np = np.array([seq]) # Shape (1, maxlen)
            item_indices_np = np.array(items_to_score) # Shape (len(items_to_score),)

            # predictions are logits
            # The user_id itself is not used by the current SASRec.predict method but is part of signature
            scores = model.predict(np.array([user_id]), log_seqs_np, item_indices_np) # scores for items in item_indices_np
            
            # scores is a tensor on device, move to cpu and numpy
            scores_np = scores.cpu().detach().numpy().flatten() # Shape (len(items_to_score),)

            # Get top N items
            # Create pairs of (score, item_id) to sort
            scored_items = list(zip(scores_np, items_to_score))
            scored_items.sort(key=lambda x: x[0], reverse=True) # Sort by score descending
            
            top_n_recommendations = [item_id for score, item_id in scored_items[:args.top_n]]
            recommendations[user_id] = top_n_recommendations

            # For now, just print (can be changed to save to file)
            # print(f"User {user_id}: {top_n_recommendations}")

        print("\n--- Top-N Recommendations ---")
        for user_id, recs in recommendations.items():
            if recs: # Only print if there are recommendations
                 print(f"User {user_id}: {recs}")
        print("--- End of Recommendations ---")
        # Here you could save `recommendations` to a file, e.g., JSON or CSV
        # import json
        # with open(os.path.join(args.dataset + '_' + args.train_dir, 'recommendations.json'), 'w') as f_recs:
        #     json.dump(recommendations, f_recs)
        # print(f"Recommendations saved to {os.path.join(args.dataset + '_' + args.train_dir, 'recommendations.json')}")

    elif args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args) # Returns (NDCG, P, R)
        print('test (NDCG@10: %.4f, P@10: %.4f, R@10: %.4f)' % (t_test[0], t_test[1], t_test[2]))
    
    else: # Training mode
        num_batch = (len(user_train) - 1) // args.batch_size + 1
        cc = 0.0
        for u in user_train:
            cc += len(user_train[u])
        print('average sequence length: %.2f' % (cc / len(user_train)))
        
        f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
        f.write('epoch (val_ndcg@10, val_p@10, val_r@10) (test_ndcg@10, test_p@10, test_r@10)\n')
        
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
        
        for name, param in model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass # just ignore those failed init layers

        model.pos_emb.weight.data[0, :] = 0
        model.item_emb.weight.data[0, :] = 0
        
        model.train() # enable model training
        
        epoch_start_idx = 1
        if args.state_dict_path is not None: # This check is already done above and model loaded
            # Logic for resuming epoch_start_idx if model was loaded for continued training
            # This part should only run if NOT in generate_recommendations mode
            if not args.generate_recommendations:
                try:
                    # model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device))) # Already loaded
                    tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
                    epoch_start_idx = int(tail[:tail.find('.')]) + 1
                except: 
                    print('Failed to parse epoch from state_dict_path for training resumption.')
                    # import pdb; pdb.set_trace() # No pdb in non-interactive
        
        bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

        best_val_ndcg, best_val_precision, best_val_recall = 0.0, 0.0, 0.0
        best_test_ndcg, best_test_precision, best_test_recall = 0.0, 0.0, 0.0
        T = 0.0
        t0 = time.time()
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

            if epoch % 20 == 0:
                model.eval()
                t1 = time.time() - t0
                T += t1
                print('Evaluating', end='')
                t_test = evaluate(model, dataset, args) # (NDCG, P, R)
                t_valid = evaluate_valid(model, dataset, args) # (NDCG, P, R)
                print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, P@10: %.4f, R@10: %.4f), test (NDCG@10: %.4f, P@10: %.4f, R@10: %.4f)'
                        % (epoch, T, t_valid[0], t_valid[1], t_valid[2], t_test[0], t_test[1], t_test[2]))

                # Update best metrics condition. Let's prioritize recall for saving, similar to old HR.
                # Or a combination, e.g. if val_recall improves or if val_ndcg improves.
                # For now, let's use: if new validation recall or ndcg is better, or if new test recall or ndcg is better.
                if t_valid[2] > best_val_recall or t_valid[0] > best_val_ndcg or \
                   t_test[2] > best_test_recall or t_test[0] > best_test_ndcg:
                    
                    best_val_ndcg = max(t_valid[0], best_val_ndcg)
                    best_val_precision = max(t_valid[1], best_val_precision) # Keep track of best P too
                    best_val_recall = max(t_valid[2], best_val_recall)
                    
                    best_test_ndcg = max(t_test[0], best_test_ndcg)
                    best_test_precision = max(t_test[1], best_test_precision)
                    best_test_recall = max(t_test[2], best_test_recall)
                    
                    folder = args.dataset + '_' + args.train_dir
                    fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                    fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                    torch.save(model.state_dict(), os.path.join(folder, fname))

                f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()
                t0 = time.time()
                model.train()
        
            if epoch == args.num_epochs:
                folder = args.dataset + '_' + args.train_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))
        
        f.close()
        sampler.close()
    
    print("Done")
