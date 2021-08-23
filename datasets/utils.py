import torch
import numpy as np


def split_patches(imgs, H, W, P):
    """
    Split image into patches. Apply for batches
    """
    assert int(W*H / (P*P)) * (P*P) == W*H, "Not divisible"
    num_patches_w = int(W/P)
    num_patches_h = int(H/P)
    num_channels = imgs.shape[1]
    batch_patches = []

    for img in imgs:
        patches = []
        for i in range(1, num_patches_h+1):
            for j in range(1, num_patches_w+1):
                patches.append(img[:, P*(i-1):P*i, P*(j-1):P*j].clone())
        patches = torch.stack(patches).reshape(-1, num_channels*P*P)
        batch_patches.append(patches)
    batch_patches = torch.stack(batch_patches, dim=0)
    return batch_patches  

def make_text_feature_batch(features,  pad_token=0):
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