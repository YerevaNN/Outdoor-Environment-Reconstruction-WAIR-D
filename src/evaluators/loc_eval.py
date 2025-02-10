import os

import numpy as np
from tqdm import tqdm

from src.datamodules.datasets import WAIRDDatasetImages


class LocalizationEvaluation:
    
    def __init__(self, prediction_path, dataset):
        assert isinstance(dataset, WAIRDDatasetImages), "LocalizationEvaluation works only with WAIRDDatasetImages"
        
        self._prediction_path = prediction_path
        self._dataset = dataset
        self._rmses = None
        self._is_los_list = None
    
    @property
    def rmses(self):
        if self._rmses is None:
            self._rmses, self._is_los_list = self.get_rmses()
        return self._rmses
    
    def get_rmses(self) -> tuple[np.ndarray, np.ndarray]:
        rmses = []
        is_los_list = []
        for i, batch in tqdm(enumerate(self._dataset), total=len(self._dataset)):
            pred_path = os.path.join(self._prediction_path, f"{i}.npz")
            input_image, supervision_image, ue_location, image_size, is_los = batch
            out = list(np.load(pred_path, allow_pickle=True).values())[0]
            scale = image_size / out.shape[0]
            # TODO add support for regression models
            max_ind = out.flatten().argmax()
            ue_location_pred = np.array([max_ind % max(out.shape), max_ind // max(out.shape)])
            rmse = float(((ue_location_pred - ue_location) ** 2).sum() ** 0.5) * scale
            rmses.append(rmse)
            is_los_list.append(is_los)
        rmses = np.array(rmses)
        is_los_list = np.array(is_los_list)
        return rmses, is_los_list
    
    def get_rmse_all_los_nlos(self) -> tuple[float, float, float]:
        return self.rmses.mean(), self.rmses[self._is_los_list].mean(), self.rmses[~self._is_los_list].mean()
    
    def get_accuracy(self, allowable_errors: int) -> list[float]:
        allowable_errors = np.array(allowable_errors)[np.newaxis]
        return ((self.rmses[:, np.newaxis] < allowable_errors).sum() / len(self._is_los_list)).tolist()
    
    def get_accuracy_all_los_nlos(self, allowable_errors: int) -> tuple[list[float], list[float], list[float]]:
        allowable_errors = np.array(allowable_errors)[np.newaxis]
        rmses = self.rmses[:, np.newaxis]
        return (
            ((rmses < allowable_errors).sum(axis=0) / len(self._is_los_list)).tolist(),
            ((rmses[self._is_los_list] < allowable_errors).sum(axis=0) / sum(self._is_los_list)).tolist(),
            ((rmses[~self._is_los_list] < allowable_errors).sum(axis=0) / sum(~self._is_los_list)).tolist()
        )
