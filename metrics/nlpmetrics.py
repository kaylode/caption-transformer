import os
import torch
import json
import numpy as np
from tqdm import tqdm
from .metrictemplate import TemplateMetric
from pycocotools.coco import COCO
from .pycocoevalcap.eval import COCOEvalCap

"""
https://github.com/salaniz/pycocoevalcap
"""

"""
GT format
annotation{
  "id": int, 
  "image_id": int, 
  "caption": str,
}

Result format
[{
    "image_id": int, 
    "caption": str,
}]
"""

def _eval(gt_json_path, pred_json_path, image_ids=None, metrics_list=['bleu', "meteor", 'rouge', 'cider', 'spice']):

    coco_gt = COCO(gt_json_path)
    
    if image_ids is None:
        image_ids = coco_gt.getImgIds()

    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    coco_eval = COCOEvalCap(coco_gt, coco_pred)
    coco_eval.params['image_id'] = image_ids

    # Set evaluation metrics
    coco_eval.setMetrics(metrics_list)
    coco_eval.evaluate()

    # create output dictionary
    stats = {}
    for metric, score in coco_eval.eval.items():
        stats[metric] = score

    # Get average metric score
    if 'bleu' in metrics_list:
        bleu_count = 0
        bleu_score = 0
        for metric in stats.keys():
            if 'Bleu' in metric:
                bleu_count += 1
                bleu_score += stats[metric]
        stats['BLEU'] = bleu_score/bleu_count

    return stats


class NLPMetrics(TemplateMetric):
    def __init__(
            self,
            dataloader, 
            max_samples = None,
            metrics_list=['bleu', "meteor", 'rouge', 'cider', 'spice'],
            decimals = 5):

        self.dataloader = dataloader
        self.max_samples = max_samples
        self.decimals = decimals
        self.filepath = f'results/text_results.json'
        self.gt_filepath = self.dataloader.dataset.ann_path
        self.metrics_list = metrics_list
        self.reset()

        if not os.path.exists('results'):
            os.mkdir('results')
            
    def reset(self):
        self.model = None

    def update(self, model):
        self.model = model
        self.model.eval()

    def compute(self):
        result_dict = []

        with torch.no_grad():
            if self.max_samples is not None:
                total_iter = min(len(self.dataloader)-1, int(self.max_samples/self.dataloader.batch_size))
            else:
                total_iter = len(self.dataloader)-1

            with tqdm(total=total_iter) as pbar:
                for idx, batch in enumerate(self.dataloader):
                    if idx > total_iter:
                        break
                    
                    image_ids = batch['image_ids']
                    preds = self.model.inference_step(batch, self.dataloader.tokenizer)

                    for image_id, pred in zip(image_ids, preds):
                            
                        result_dict.append({
                            'image_id': image_id,
                            'caption': pred
                        })
                                                    
                    pbar.update(1)

        if not len(result_dict):
            return False

        # write output
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        json.dump(result_dict, open(self.filepath, 'w'), indent=4)

        return True

    def value(self):
        self.compute()
        stats = _eval(
            self.gt_filepath, self.filepath, None, self.metrics_list)

        # Round up
        for key, value in stats.items():
            stats[key] = np.round(float(value), self.decimals)
        
        return stats

    def __str__(self):
        return f'{self.value()}'

    def __len__(self):
        return len(self.dataloader)

