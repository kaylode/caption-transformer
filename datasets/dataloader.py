from .dataset import ImageTextSet
from .cocoset import CocoDataset
from torch.utils.data import DataLoader
from torchtext.legacy.data import BucketIterator


class EqualLengthTextLoader(BucketIterator):
    """
    Use BucketIterator to make texts of same length into batch
    """
    def __init__(self, 
                batch_size,
                root_dir,
                ann_path,
                tokenizer,
                image_size,
                patch_size,
                keep_ratio,
                device,
                type,
                **kwargs):
       
        self.dataset = CocoDataset(
                root_dir=root_dir, ann_path=ann_path, 
                tokenizer=tokenizer, image_size=image_size, 
                keep_ratio=keep_ratio, patch_size=patch_size, type=type)

        self.stokenizer = self.dataset.tokenizer
        self.collate_fn = self.dataset.collate_fn
        
        super(EqualLengthTextLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            device=device, 
            sort_key=lambda x: len(x['text']),
            repeat=True, # Repeat the iterator for multiple epochs.
            sort=False,  # Sort all examples in data using `sort_key`.
            shuffle=True, # Shuffle data on each epoch run.
            sort_within_batch=True) # Use `sort_key` to sort examples in each batch.


class RawTextLoader(DataLoader):
    """
    Use DataLoader to make texts into batch
    """
    def __init__(self, 
                batch_size,
                root_dir,
                ann_path,
                tokenizer,
                image_size,
                patch_size,
                keep_ratio,
                type,
                **kwargs):
       
        self.dataset = CocoDataset(
                root_dir=root_dir, ann_path=ann_path, 
                tokenizer=tokenizer, image_size=image_size, 
                keep_ratio=keep_ratio, patch_size=patch_size, type=type)

        self.tokenizer = self.dataset.tokenizer
        self.collate_fn = self.dataset.collate_fn
        
        super(RawTextLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_fn)