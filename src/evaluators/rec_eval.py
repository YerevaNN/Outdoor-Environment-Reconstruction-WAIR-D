import os

import numpy as np
import torch
from tqdm import tqdm

from src.datamodules.datasets import WAIRDDatasetRVarChannelsSequences
from src.utils import chamfer_dist, hausdorff_dist, iog_score, iop_score, iou_score


class ReconstructionEvaluation:
    
    def __init__(self, prediction_path, dataset):
        assert isinstance(
            dataset, WAIRDDatasetRVarChannelsSequences
        ), "ReconstructionEvaluation works only with WAIRDDatasetRVarChannelsSequences"
        
        self._prediction_path = prediction_path
        self._dataset = dataset
        self.ious = []
        self.iogs = []
        self.iops = []
        self.hausdorffs = []
        self.chamfers = []
    
    def get_all_metrics(self) -> tuple[float, float, float, float, float]:
        if not self.ious:
            for i, batch in tqdm(enumerate(self._dataset), total=len(self._dataset)):
                pred_path = os.path.join(self._prediction_path, f"{i}.npz")
                input_img, map_img, sequences, image_size = batch
                out = torch.Tensor([[list(np.load(pred_path, allow_pickle=True).values())[0]]])
                map_img = torch.Tensor([map_img])
                scale = image_size / out.shape[-1]
                
                self.ious.append(iou_score(out, map_img))
                self.iogs.append(iog_score(out, map_img))
                self.iops.append(iop_score(out, map_img))
                self.hausdorffs.append(scale * hausdorff_dist(out, map_img))
                self.chamfers.append(scale * chamfer_dist(out, map_img))
        
        return (
            np.array(self.ious).mean(),
            np.array(self.iogs).mean(),
            np.array(self.iops).mean(),
            np.array(self.hausdorffs).mean(),
            np.array(self.chamfers).mean()
        )
