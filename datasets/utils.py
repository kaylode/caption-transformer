import sys
import csv
import torch
import base64
import numpy as np

csv.field_size_limit(sys.maxsize)

def make_feature_batch(features,  pad_token=0):
    """
    List of features,
    each feature is [K, model_dim] where K is number of objects of each image
    This function pad max length to each feature and tensorize, also return the masks
    """

    # Find maximum length
    max_len=0
    for feat in features:
        feat_len = feat.shape[0]
        max_len = max(max_len, feat_len)
    
    # Init batch
    batch_size = len(features)
    batch = np.ones((batch_size, max_len))
    batch *= pad_token
        
    # Copy data to batch
    for i, feat in enumerate(features):
        feat_len = feat.shape[0]
        batch[i, :feat_len] = feat

    batch = torch.from_numpy(batch).type(torch.float32)
    return batch  


def create_masks(features, pad_token=0, is_tgt_masking=False):
    """
    Create masks from features
    """
    masks = (features != pad_token)
    if is_tgt_masking:
        size = features.size(1)
        nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        nopeak_mask = torch.from_numpy(nopeak_mask) == 0
        masks = masks.unsqueeze(1) & nopeak_mask
    
    return masks    

def efficient_iterrows(filename):
    """
    Initiate generator from tsv file in chunks
    """
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile, delimiter='\t')
        # print(next(datareader))
        yield next(datareader)  # yield the header row
        for row in datareader:
            yield row

def np_from_b64(b64_str, dtype):
    """
    Decode base64 from numpy array
    """
    decoded_arr = base64.b64decode(b64_str.encode())
    decoded_arr = np.frombuffer(decoded_arr, dtype=dtype)
    return decoded_arr

def decode_tsv(filename):
    """
    Decode tsv file into features dataset
    TSV generated from: https://github.com/airsplay/py-bottom-up-attention/blob/master/demo/detectron2_mscoco_proposal_maxnms.py
    """
    from tqdm import tqdm

    array = []
    for row in tqdm(efficient_iterrows(filename)):
        item = {
            "img_id": row[0],
            "img_h": int(row[1]),
            "img_w": int(row[2]), 
            "objects_id": np_from_b64(row[3], dtype=np.int64),  # int64
            "objects_conf": np_from_b64(row[4], dtype=np.float32),  # float32
            "attrs_id": np_from_b64(row[5], dtype=np.int64),  # int64
            "attrs_conf": np_from_b64(row[6], dtype=np.float32),  # float32
            "num_boxes": int(row[7]),
            "boxes": np_from_b64(row[8], dtype=np.float32),  # float32
            "features": np_from_b64(row[9], dtype=np.float32)  # float32
        }
        array.append(item)

    return array