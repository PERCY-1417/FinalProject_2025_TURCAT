import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import os

# Number of items for validation and test sets
NUM_VALID_ITEMS = 10
NUM_TEST_ITEMS = 10 # Users will have 2 items in their test set
K_EVAL = 10 # Evaluation cutoff for @K metrics

DATA_PATH = "data/"
DATA_PATH = "data_final_project/KuaiRec 2.0/sas_rec_data/"

def build_index(dataset_name):

    ui_mat = np.loadtxt(DATA_PATH + '%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    # Define a function to sample a single user
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        # Ensure the user has more than one interaction
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        # Initialize numpy arrays for sequence, positive, and negative samples
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        # The target item is the last item in the user's sequence
        nxt = user_train[uid][-1]
        # Index for filling the sequence arrays, starting from the end
        idx = maxlen - 1

        # Create a set of the user's interactions for negative sampling
        ts = set(user_train[uid])
        # Iterate through the user's sequence in reverse order, excluding the last item
        # The goal is to create a sequence of interactions, a positive sample (the next item), and a negative sample (an item the user didn't interact with) for each position in the sequence.
        for i in reversed(user_train[uid][:-1]):
            # Store the item in the sequence
            seq[idx] = i
            # Store the next item as the positive sample
            pos[idx] = nxt
            # Sample a negative item if the next item is not padding (0)
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            # Update the next item and index
            nxt = i
            idx -= 1
            # Stop if the index reaches -1
            if idx == -1: break

        # Return the user ID, sequence, positive samples, and negative samples
        return (uid, seq, pos, neg)

    # Set the random seed for reproducibility
    np.random.seed(SEED)
    # Create an array of user IDs
    uids = np.arange(1, usernum+1, dtype=np.int32)
    # Initialize a counter
    counter = 0
    # Run indefinitely to generate batches
    while True:
        # Shuffle user IDs every time the counter is a multiple of usernum
        if counter % usernum == 0:
            np.random.shuffle(uids)
        # Initialize a list to store a batch of samples
        one_batch = []
        # Generate a batch of samples
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        # Put the batch into the result queue
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        # print(f'running next_batch')
        # print(f'self.result_queue: {self.result_queue}')
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def save_split_to_file(split_dict, filename, out_dir):
    with open(os.path.join(out_dir, filename), "w") as f:
        for user, items in split_dict.items():
            for item in items:
                f.write(f"{user} {item}\n")

def data_partition(fname, save_files=True, out_dir=None):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(DATA_PATH + '%s.txt' % fname, 'r')
    for line in f:
        u, i, e = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    MIN_INTERACTIONS_FOR_FULL_SPLIT = 1 + NUM_VALID_ITEMS + NUM_TEST_ITEMS # Min 1 for train

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < MIN_INTERACTIONS_FOR_FULL_SPLIT:
            # Not enough interactions for a full train/valid/test split according to NUM_VALID_ITEMS and NUM_TEST_ITEMS
            # Fallback: use all for training, empty valid/test. Or a simpler split.
            # For now, let's keep it simple: if not enough for chosen K, they mostly go to train.
            # This part might need more nuanced handling depending on desired behavior for sparse users.
            if nfeedback >= 3: # Original minimum: 1 train, 1 val, 1 test
                 user_train[user] = User[user][:-2]
                 user_valid[user] = [User[user][-2]]
                 user_test[user] = [User[user][-1]]
            elif nfeedback == 2:
                 user_train[user] = [User[user][0]]
                 user_valid[user] = [User[user][1]]
                 user_test[user] = []
            else: # nfeedback == 1
                 user_train[user] = User[user]
                 user_valid[user] = []
                 user_test[user] = []
        else:
            # Full split based on NUM_VALID_ITEMS and NUM_TEST_ITEMS
            # Example: if NUM_VALID_ITEMS=1, NUM_TEST_ITEMS=2
            # train: items up to last 3
            # valid: 3rd to last item
            # test: last 2 items
            user_train[user] = User[user][:-(NUM_VALID_ITEMS + NUM_TEST_ITEMS)]
            user_valid[user] = User[user][-(NUM_VALID_ITEMS + NUM_TEST_ITEMS) : -NUM_TEST_ITEMS]
            user_test[user] = User[user][-NUM_TEST_ITEMS:]
    
    # Save splits to files if requested
    if save_files and out_dir is not None:
        save_split_to_file(user_train, "train.txt", out_dir)
        save_split_to_file(user_valid, "validation.txt", out_dir)
        save_split_to_file(user_test, "test.txt", out_dir)
    
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    total_ndcg_at_k = 0.0
    total_precision_at_k = 0.0
    total_recall_at_k = 0.0
    evaluated_users = 0.0

    users_to_evaluate = range(1, usernum + 1)
    if usernum > 10000: # Sample users if too many
        users_to_evaluate = random.sample(range(1, usernum + 1), 10000)

    users_to_evaluate = list(test.keys()) # Only evaluate users with test items

    for u in users_to_evaluate:
        true_positive_items = test[u]
        if len(train[u]) < 1 or not true_positive_items:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        
        # Add validation items to sequence history
        for val_item in reversed(valid[u]):
            if idx == -1: break
            seq[idx] = val_item
            idx -= 1
        
        # Add training items to sequence history
        for train_item in reversed(train[u]):
            if idx == -1: break
            seq[idx] = train_item
            idx -= 1

        rated_items = set(train[u]) | set(valid[u]) | set(true_positive_items)
        
        # Items to rank: true positives + 100 negative samples
        items_to_rank = list(true_positive_items)
        negative_samples = []
        for _ in range(100): # Number of negative samples
            neg_item = np.random.randint(1, itemnum + 1)
            while neg_item in rated_items:
                neg_item = np.random.randint(1, itemnum + 1)
            negative_samples.append(neg_item)
            rated_items.add(neg_item) # Add to rated to avoid re-sampling same negative
        
        items_to_rank.extend(negative_samples)
        
        # Get predictions from model
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], items_to_rank]])
        predictions = predictions[0] # predictions for user u

        # Get top K recommended items
        top_k_indices = predictions.argsort()[:K_EVAL]
        recommended_items = [items_to_rank[i] for i in top_k_indices]

        # Calculate metrics
        hits_at_k = 0
        dcg_at_k = 0.0
        true_positive_set = set(true_positive_items)

        for i, rec_item in enumerate(recommended_items):
            if rec_item in true_positive_set:
                hits_at_k += 1
                dcg_at_k += 1.0 / np.log2(i + 2) # rank is i+1

        idcg_at_k = 0.0
        for i in range(min(len(true_positive_items), K_EVAL)):
            idcg_at_k += 1.0 / np.log2(i + 2)
        
        ndcg_at_k = dcg_at_k / idcg_at_k if idcg_at_k > 0 else 0.0
        precision_at_k = hits_at_k / K_EVAL
        recall_at_k = hits_at_k / len(true_positive_items) if len(true_positive_items) > 0 else 0.0
        
        total_ndcg_at_k += ndcg_at_k
        total_precision_at_k += precision_at_k
        total_recall_at_k += recall_at_k
        evaluated_users += 1

        if evaluated_users % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    avg_ndcg = total_ndcg_at_k / evaluated_users if evaluated_users > 0 else 0.0
    avg_precision = total_precision_at_k / evaluated_users if evaluated_users > 0 else 0.0
    avg_recall = total_recall_at_k / evaluated_users if evaluated_users > 0 else 0.0
    
    return avg_ndcg, avg_precision, avg_recall


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    total_ndcg_at_k = 0.0
    total_precision_at_k = 0.0
    total_recall_at_k = 0.0
    evaluated_users = 0.0

    users_to_evaluate = range(1, usernum + 1)
    if usernum > 10000: # Sample users if too many
        users_to_evaluate = random.sample(range(1, usernum + 1), 10000)

    for u in users_to_evaluate:
        true_positive_items = valid[u] # This is a list
        if len(train[u]) < 1 or not true_positive_items:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for train_item in reversed(train[u]): # History for validation is only train data
            if idx == -1: break
            seq[idx] = train_item
            idx -= 1

        rated_items = set(train[u]) | set(true_positive_items)
        
        items_to_rank = list(true_positive_items)
        negative_samples = []
        for _ in range(100): # Number of negative samples
            neg_item = np.random.randint(1, itemnum + 1)
            while neg_item in rated_items:
                neg_item = np.random.randint(1, itemnum + 1)
            negative_samples.append(neg_item)
            rated_items.add(neg_item)

        items_to_rank.extend(negative_samples)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], items_to_rank]])
        predictions = predictions[0]

        top_k_indices = predictions.argsort()[:K_EVAL]
        recommended_items = [items_to_rank[i] for i in top_k_indices]

        hits_at_k = 0
        dcg_at_k = 0.0
        true_positive_set = set(true_positive_items)

        for i, rec_item in enumerate(recommended_items):
            if rec_item in true_positive_set:
                hits_at_k += 1
                dcg_at_k += 1.0 / np.log2(i + 2)

        idcg_at_k = 0.0
        for i in range(min(len(true_positive_items), K_EVAL)): # true_positive_items is often 1 for valid
            idcg_at_k += 1.0 / np.log2(i + 2)
        
        ndcg_at_k = dcg_at_k / idcg_at_k if idcg_at_k > 0 else 0.0
        precision_at_k = hits_at_k / K_EVAL
        recall_at_k = hits_at_k / len(true_positive_items) if len(true_positive_items) > 0 else 0.0
        
        total_ndcg_at_k += ndcg_at_k
        total_precision_at_k += precision_at_k
        total_recall_at_k += recall_at_k
        evaluated_users += 1
        
        if evaluated_users % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    avg_ndcg = total_ndcg_at_k / evaluated_users if evaluated_users > 0 else 0.0
    avg_precision = total_precision_at_k / evaluated_users if evaluated_users > 0 else 0.0
    avg_recall = total_recall_at_k / evaluated_users if evaluated_users > 0 else 0.0

    return avg_ndcg, avg_precision, avg_recall

def get_user_item_counts(fname):
    """
    Quickly scan a dataset file to get the max user and item IDs.
    Returns (usernum, itemnum)
    """
    max_user = 0
    max_item = 0
    with open(DATA_PATH + '%s.txt' % fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            if u > max_user:
                max_user = u
            if i > max_item:
                max_item = i
    return max_user, max_item