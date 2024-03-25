import os
import json
import h5py


def convert_h5_to_json_standard(h5_path, json_save_path):
    # read h5 file
    h5_file = h5py.File(h5_path, "r")
    # get keys
    keys = list(h5_file.keys())
    assert 'masks' in keys, "masks not in keys"

    data = h5_file['masks']

    print("parsing h5 file...")
    dataset = [eval(data[i].decode('utf-8')) for i in range(len(data))]

    # convert dataset to json compatitible
    for sample in dataset:
        masks = sample['masks']
        for mask in masks:
            seg = mask['segmentation']
            seg['counts'] = seg['counts'].decode()

    # save to json
    print("saving to json...")
    with open(json_save_path, 'w') as f:
        json.dump(dataset, f)
    
    print("saved to {}".format(json_save_path))

def convert_h5_to_json_reasonSeg(base_dir, json_save_path):
    splits = ['train', 'val']
    for split in splits:
        h5_path = os.path.join(base_dir, "ReasonSeg", split, "masks.h5")
        json_save_path = os.path.join(base_dir, "ReasonSeg", split, "masks.json")
        convert_h5_to_json_standard(h5_path, json_save_path)

def convert_h5_to_json_coco(h5_dir, json_save_path):
    # coco have 8 splits
    dataset = []
    for i in range(8):
        # if i == 2:
        #     break
        h5_path = os.path.join(h5_dir, "coco_split{}.h5".format(i))
        # read h5 file
        h5_file = h5py.File(h5_path, "r")
        # get keys
        keys = list(h5_file.keys())
        assert 'masks' in keys, "masks not in keys"
        data = h5_file['masks']

        print("Parsing split {}...".format(i))
        dataset_split = [eval(data[i].decode('utf-8')) for i in range(len(data))]
        dataset.extend(dataset_split)

    
    # convert dataset to json compatitible
    for sample in dataset:
        masks = sample['masks']
        for mask in masks:
            seg = mask['segmentation']
            seg['counts'] = seg['counts'].decode()
    
    # save to json
    print("saving to json...")
    with open(json_save_path, 'w') as f:
        json.dump(dataset, f)




def main():
    base_dir = "/cluster/home/leikel/junchi/processed_data/"

    dataset_names = ["ade20k", "coco2014", "coco2017", "reason_seg", "saiapr", "voc2010"]
    # dataset_names = ["reason_seg"]
    # dataset_names = ["ade20k"]
    # dataset_names = ["coco2014", "coco2017"]

    for dataset_name in dataset_names:
        print("processing {}".format(dataset_name))
        dataset_dir = os.path.join(base_dir, dataset_name)
        h5_path = os.path.join(dataset_dir, "masks.h5")
        json_save_path = os.path.join(dataset_dir, "masks.json")
        if 'coco' in dataset_name:
            convert_h5_to_json_coco(dataset_dir, json_save_path)
        elif dataset_name == "reason_seg":
            convert_h5_to_json_reasonSeg(dataset_dir, json_save_path)
        else:
            convert_h5_to_json_standard(h5_path, json_save_path)
            

if __name__ == "__main__":
    main()
    
