import pandas as pd
import os
import sys
from collections import defaultdict

# Usage: python prepare_data.py small_matrix
if len(sys.argv) < 2:
    print("Usage: python prepare_data.py <dataset_name>")
    sys.exit(1)

dataset_name = sys.argv[1]
print(f"Preparing data for dataset: {dataset_name}")

# Load the interaction data
csv_path = f'data_final_project/KuaiRec 2.0/data/{dataset_name}.csv'
print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

print("First few rows of the dataset:")
print(df.head())

# sort the dataframe by user_id and timestamp
print("Sorting dataframe by user_id and timestamp...")
df.sort_values(by=['user_id', 'timestamp'], inplace=True)

# Build user -> [video_id1, video_id2, ...] dict
user_sequences = defaultdict(list)
for row in df.itertuples():
    user_sequences[row.user_id].append(row.video_id)

sas_rec_data_dir = "data_final_project/KuaiRec 2.0/sas_rec_data/"
os.makedirs(sas_rec_data_dir, exist_ok=True)
print(f"Saving processed files to: {sas_rec_data_dir}")

# Save all user-item interactions to a single file WITH liked tag and remapped user IDs
user_id_map = {uid: idx+1 for idx, uid in enumerate(user_sequences.keys())}
print(f"Writing remapped user IDs to {sas_rec_data_dir}{dataset_name}.txt ...")
with open(sas_rec_data_dir + f"{dataset_name}.txt", "w") as f:
    for row in df.itertuples():
        new_user = user_id_map[row.user_id]
        item = row.video_id
        liked = 1 if row.watch_ratio > 2.0 else 0
        f.write(f"{new_user} {item} {liked}\n")
print("Done writing remapped file.")

# Save all user-item interactions to a single file WITHOUT remapping user IDs, WITH liked tag
print(f"Writing original user IDs to {sas_rec_data_dir}{dataset_name}_no_remapping.txt ...")
with open(sas_rec_data_dir + f"{dataset_name}_no_remapping.txt", "w") as f:
    for row in df.itertuples():
        user = row.user_id
        item = row.video_id
        liked = 1 if row.watch_ratio > 2.0 else 0
        f.write(f"{user} {item} {liked}\n")
print("Done writing no-remapping file.")