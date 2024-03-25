import h5py
import os
import pickle

from typing import List, Dict


def read_mask_list(path: str):
    h5file = h5py.File(path, "r")
    dataset = [h5file["masks"][i].decode("utf-8") for i in range(len(h5file["masks"]))]
    dataset_restored = [eval(data) for data in dataset]

    return dataset_restored

def build_sam_mask_dict(mask_list: List[Dict]):
    sam_mask_dict = {}
    for i, sample in enumerate(mask_list):
        sample_name = sample["image"]
        sam_mask_dict[sample_name] = i

    return sam_mask_dict

def main():
    mask_dir = "/cluster/home/leikel/junchi/processed_data/reason_seg/ReasonSeg/"
    split = ["train", "val"]

    for s in split:
        mask_file = os.path.join(mask_dir, s, "masks.h5")
        mask_list = read_mask_list(mask_file)
        sam_mask_dict = build_sam_mask_dict(mask_list)

        print("len(sam_mask_dict): ", len(sam_mask_dict))

        # save sam_mask_dict as pickle file
        with open(os.path.join(mask_dir, s, "sam_mask_index_dict.pkl"), "wb") as f:
            pickle.dump(sam_mask_dict, f)
        

if __name__ == "__main__":
    main()