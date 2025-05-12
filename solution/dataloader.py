import os
import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def data_partition(data_dir):
    user_train = {}
    user_valid = {}
    user_test = {}

    def read_sequence(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                user_part, items_part = line.strip().split('\t')
                user_id = int(user_part)
                items = [int(x) for x in items_part.split(',') if x]
                yield user_id, items

    # Read train data
    for user, items in read_sequence(os.path.join(data_dir, 'train.txt')):
        user_train[user] = items

    # Read valid data
    for user, items in read_sequence(os.path.join(data_dir, 'valid.txt')):
        user_valid[user] = items

    # Read test data
    for user, items in read_sequence(os.path.join(data_dir, 'test.txt')):
        user_test[user] = items

    usernum = max(user_train.keys()) + 1

    # Find max item ID across all sets
    itemnum = 0
    for dataset in (user_train, user_valid, user_test):
        for items in dataset.values():
            if items:
                itemnum = max(itemnum, max(items))
    itemnum += 1  # account for 0-padding

    print(f"[data_partition] Users: {usernum}, Items: {itemnum}")
    return user_train, user_valid, user_test, usernum, itemnum

def build_index(data_dir):
    file_path = os.path.join(data_dir, 'train.txt')
    u2i_index = defaultdict(list)
    i2u_index = defaultdict(list)

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            user_part, items_part = line.strip().split('\t')
            user_id = int(user_part)
            items = [int(x) for x in items_part.split(',') if x]
            for item in items:
                u2i_index[user_id].append(item)
                i2u_index[item].append(user_id)

    # Convert defaultdicts to lists for compatibility
    max_user = max(u2i_index.keys())
    max_item = max(i2u_index.keys())
    u2i_list = [[] for _ in range(max_user + 1)]
    i2u_list = [[] for _ in range(max_item + 1)]
    for u, items in u2i_index.items():
        u2i_list[u] = items
    for i, users in i2u_index.items():
        i2u_list[i] = users

    return u2i_list, i2u_list

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(usernum, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(usernum), 10000)
    else:
        users = range(usernum)

    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum)
            while t in rated: t = np.random.randint(1, itemnum)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

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
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user