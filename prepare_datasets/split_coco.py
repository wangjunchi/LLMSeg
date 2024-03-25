'''
Read the file list of coco training set, and split it into 8 parts.
'''

import os

# dataset_path = "/cluster/scratch/leikel/junchi/lisa_dataset/coco/train2017"
dataset_path = "/cluster/scratch/leikel/junchi/lisa_dataset/refer_seg/images/mscoco/images/train2014"
output_path = "/cluster/home/leikel/junchi/processed_data/coco2014"

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

files = os.listdir(dataset_path)
# filter out non-image files
files = [file for file in files if file.endswith(".jpg")]

print("Total number of files: {}".format(len(files)))

# split into 8 parts
num_parts = 8

part_cnt = [0 for i in range(num_parts)]

for i in range(num_parts):
    part_files = files[i::num_parts]
    part_files.sort()
    part_cnt[i] = len(part_files)
    with open(os.path.join(output_path, "part_{}.txt".format(i)), "w") as f:
        for file in part_files:
            f.write(file + "\n")

assert sum(part_cnt) == len(files), "Number of files in parts does not match total number of files."

print("Done.")