import pandas as pd

# Load the interaction data
df = pd.read_csv('data_final_project/KuaiRec 2.0/data/small_matrix.csv')

# Display the first few rows to understand its structure
print(df.head())

from collections import defaultdict

# sort the dataframe by user_id and timestamp
df.sort_values(by=['user_id', 'timestamp'], inplace=True)

# Build user -> [video_id1, video_id2, ...] dict
user_sequences = defaultdict(list)

for row in df.itertuples():
    user_sequences[row.user_id].append(row.video_id)

# Preview one user's sequence
for uid, vids in list(user_sequences.items())[:1]:
    print(f"User {uid} : {vids}")
print(f"Number of unique userssequence: {len(user_sequences)}")

import os

sas_rec_data_dir = "data_final_project/KuaiRec 2.0/sas_rec_data/"
# Create the folder if it does not exist
os.makedirs(sas_rec_data_dir, exist_ok=True)

with open(sas_rec_data_dir + "sasrec_sequences_small_matrix.txt", "w") as f:
    for sequence in user_sequences.values():
        f.write(" ".join(map(str, sequence)) + "\n")


# Save all user-item interactions to a single file
user_id_map = {uid: idx+1 for idx, uid in enumerate(user_sequences.keys())}

with open(sas_rec_data_dir + "small_matrix.txt", "w") as f:
    for user, items in user_sequences.items():
        new_user = user_id_map[user]
        for item in items:
            f.write(f"{new_user} {item}\n")

# Save all user-item interactions to a single file WITHOUT remapping user IDs
with open(sas_rec_data_dir + "small_matrix_no_remapping.txt", "w") as f:
    for user, items in user_sequences.items():
        for item in items:
            f.write(f"{user} {item}\n")


