import os
from dataclasses import dataclass
from itertools import chain, combinations

import numpy as np
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from skimage.transform import resize
from sklearn.mixture import GaussianMixture

from src.prepare_data import _get_env_locations_data


@dataclass
class EnvInfo:
    env: str
    img_height: float
    img_width: float
    paths1: dict[str, dict[str, np.ndarray]]
    paths2: dict[str, dict[str, np.ndarray]]
    ue_locs1: np.ndarray
    ue_locs2: np.ndarray
    bs_loc2: np.ndarray
    max_tau: float


class ValidationVisualize:
    
    def __init__(self, raw_data_path: str, predictions_path: str, baseline_radius: float):
        scenario2 = "scenario_2"
        self.scenario2_path = os.path.join(os.path.dirname(raw_data_path), scenario2)
        self.scenario1_path = os.path.join(raw_data_path)
        self.environments = os.listdir(self.scenario2_path)
        self.predictions_path = predictions_path
        self.baseline_radius = baseline_radius
    
    @staticmethod
    def dissimilarity(ue_paths1: dict[str, np.ndarray], ue_paths2: dict[str, np.ndarray], max_tau: float) -> float:
        taus1 = ue_paths1["taud"]
        taus2 = ue_paths2["taud"]
        aoa1 = ue_paths1["doa"]
        aoa2 = ue_paths2["doa"]
        aod1 = ue_paths1["dod"]
        aod2 = ue_paths2["dod"]
        
        agg_axis = 0 if len(taus1) <= len(taus2) else 1
        
        aod_dists = (1 - np.cos(aod1[np.newaxis, :, 1] - aod2[:, np.newaxis, 1])) / 2
        aoa_dists = (1 - np.cos(aoa1[np.newaxis, :, 1] - aoa2[:, np.newaxis, 1])) / 2
        tau_dists = abs(taus1[np.newaxis] - taus2[:, np.newaxis]) / max_tau
        return (aod_dists + aoa_dists + tau_dists).min(axis=agg_axis).mean()
    
    @staticmethod
    def near_means(means: np.ndarray, ratio: float):
        for c1, c2, in combinations(means, 2):
            if float(((c1 - c2) ** 2).sum() ** 0.5) < 20 * ratio:
                return True
        return False
    
    @staticmethod
    def get_gmm_means(out: np.ndarray, ratio: float):
        size = out.shape
        xs = list(range(size[1])) * size[0]
        ys = list(chain.from_iterable([[i] * size[1] for i in range(size[0])]))
        z = out.flatten()
        counts = (100 * z).astype(np.uint8)
        data = list(zip(np.repeat(xs, counts), np.repeat(ys, counts)))
        if len(data) < 6:
            return np.array([])
        for n_components in range(2, 6):
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(data)
            if ValidationVisualize.near_means(gmm.means_, ratio=ratio):
                n_components -= 1
                break
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(data)
        return gmm.means_
    
    @staticmethod
    def get_similarity_expectation(
        img_height: float, img_width: float, pred: np.ndarray, ue_locs: np.ndarray, similarities: np.ndarray
    ) -> float:
        scale = max(img_height, img_width) / max(pred.shape)
        pred_ys, pred_xs = np.where(pred > (pred.max() / 2))
        pred_xs_scaled = pred_xs.astype(np.float64)
        pred_ys_scaled = pred_ys.astype(np.float64)
        pred_xs_scaled *= scale
        pred_ys_scaled *= scale
        if img_height > img_width:
            pred_xs_scaled -= (img_height - img_width) / 2
        else:
            pred_ys_scaled -= (img_width - img_height) / 2
        filter_idx = np.where(
            (pred_xs_scaled > 0) & (pred_xs_scaled < img_width) &
            (pred_ys_scaled > 0) & (pred_ys_scaled < img_height)
        )[0]
        if not filter_idx.shape[0]:
            return 0
        pred_xs = pred_xs[filter_idx]
        pred_ys = pred_ys[filter_idx]
        pred_xs_scaled = pred_xs_scaled[filter_idx]
        pred_ys_scaled = pred_ys_scaled[filter_idx]
        pred_points = np.stack([pred_xs_scaled, pred_ys_scaled], axis=-1)
        close_ues = ((pred_points[np.newaxis] - ue_locs[:, np.newaxis]) ** 2).sum(axis=-1).argmin(axis=0)
        return (similarities[close_ues] * pred[pred_xs, pred_ys]).sum() / sum(pred[pred_xs, pred_ys])
    
    @staticmethod
    def get_baseline_similarity(
        scenario1_ue: np.ndarray, ue_locs: np.ndarray, similarities: np.ndarray, radius: float
    ) -> float:
        dists = ((scenario1_ue - ue_locs) ** 2).sum(axis=-1)
        close_ues = np.where(dists < radius ** 2)[0]
        return similarities[close_ues].mean()
    
    def get_env_info(self, i):
        env = self.environments[i]
        env_path2 = os.path.join(self.scenario2_path, env)
        env_path1 = os.path.join(self.scenario1_path, env)
        _, ue_locations_df1, img_size = _get_env_locations_data(env_path1, original=True)
        ue_locs1 = ue_locations_df1[["x", "y"]].values
        # map_img = io.imread(os.path.join(env_path1, "environment.png"), as_gray=True)
        paths1 = np.load(os.path.join(env_path1, "Path.npy"), allow_pickle=True, encoding='latin1').item()
        paths2 = np.load(os.path.join(env_path2, "Path.npy"), allow_pickle=True, encoding='latin1').item()
        info_path2 = os.path.join(env_path2, "Info.npy")
        info2 = np.load(info_path2, allow_pickle=True, encoding='latin1').astype(np.float32)
        img_width, img_height = info2[:2]
        locs = info2[2:].reshape(-1, 2)
        bs_loc2 = locs[0]
        ue_locs2 = locs[1:]
        max_tau = max(
            max(chain(*(v["taud"].tolist() for k, v in paths1.items()))),
            max(chain(*(v["taud"].tolist() for k, v in paths2.items())))
        )
        return EnvInfo(
            env=env,
            img_height=img_height,
            img_width=img_width,
            paths1=paths1,
            paths2=paths2,
            ue_locs1=ue_locs1,
            ue_locs2=ue_locs2,
            bs_loc2=bs_loc2,
            max_tau=max_tau
        )
    
    @staticmethod
    def get_similarities(
        ue_idx1: int,
        paths1: dict[str, dict[str, np.ndarray]],
        paths2: dict[str, dict[str, np.ndarray]],
        max_tau: float
    ):
        dists = []
        for ue_idx2 in range(10000):
            key1 = f"bs0_ue{ue_idx1}"
            key2 = f"bs0_ue{ue_idx2:05d}"
            ue_paths1 = paths1[key1]
            ue_paths2 = paths2[key2]
            dists.append(ValidationVisualize.dissimilarity(ue_paths1, ue_paths2, max_tau))
        
        similarities = 1 - np.array(dists) / 3
        # scaling from 0 to 1
        similarities -= similarities.min()
        similarities /= similarities.max()
        return similarities
    
    @staticmethod
    def plot_similarities(fig: Figure, ax: Axes, env_info: EnvInfo, ue_idx1: int, similarities: np.ndarray):
        scenario1_ue = env_info.ue_locs1[ue_idx1]
        
        ax.set_title(f"{env_info.env}_{ue_idx1}", loc="left")
        # plt.imshow(np.flip(resize(map_img, (info2[1], info2[0])), axis=0), cmap="gray_r")
        ue_sim = ax.scatter(
            env_info.ue_locs2[:, 0], env_info.ue_locs2[:, 1], cmap="Blues_r", c=similarities, s=1
        )  # , vmin=0., vmax=1.
        fig.colorbar(ue_sim, ax=ax)
        ax.scatter(env_info.bs_loc2[0], env_info.bs_loc2[1], c="g", marker="X", label="BS", s=100)
        ax.scatter(scenario1_ue[0], scenario1_ue[1], c="b", marker="X", label="UE", s=100)
    
    @staticmethod
    def plot_prediction_contours(ax: Axes, img_height: float, img_width: float, pred: np.ndarray) -> QuadContourSet:
        img_size = int(max(img_width, img_height))
        pred_scaled = resize(pred, (img_size, img_size))
        size = pred_scaled.shape
        xs = np.array(list(range(size[1]))).astype(np.float64)
        ys = np.array(list(range(size[0]))).astype(np.float64)
        if img_width > img_height:
            ys -= (img_width - img_height) / 2
        else:
            xs -= (img_height - img_width) / 2
        xs, ys = np.meshgrid(xs, ys)
        contours = ax.contour(xs, ys, pred_scaled, levels=5, colors="orange", linewidths=1)
        return contours
    
    @staticmethod
    def plot_histogram(
        ax: Axes, env_info: EnvInfo, pred: np.ndarray, similarities: np.ndarray,
        scenario1_ue: np.ndarray, baseline_radius: float
    ):
        sns.histplot(similarities, kde=True, bins=50, shrink=.8, ax=ax, color="b")
        ax.set_xlabel("Similarity")
        similarity_expectation = ValidationVisualize.get_similarity_expectation(
            env_info.img_height, env_info.img_width, pred,
            env_info.ue_locs2, similarities
        )
        ax.axvline(
            similarity_expectation,
            c="r", alpha=0.5, label="Expected similarity"
        )
        baseline_similarity = ValidationVisualize.get_baseline_similarity(
            scenario1_ue, env_info.ue_locs2, similarities, baseline_radius
        )
        ax.axvline(
            baseline_similarity,
            c="y", alpha=0.5, label="Baseline similarity"
        )
    
    @staticmethod
    def plot_legends(heatmap_ax: Axes = None, hist_ax: Axes = None, contours: QuadContourSet = None):
        if heatmap_ax:
            handles, labels = heatmap_ax.get_legend_handles_labels()
            if contours:
                contours_legend_handles, _ = contours.legend_elements()
                handles.append(contours_legend_handles[0])
                labels.append("prediction contours")
            heatmap_ax.legend(handles, labels, loc='lower right', bbox_to_anchor=(1, 1.02), borderaxespad=0)
        if hist_ax:
            hist_ax.legend()
    
    @staticmethod
    def plot_details(ax: Axes):
        ax.set_facecolor("k")
        ax.set_axis_off()
        ax.add_artist(ax.patch)
        ax.patch.set_zorder(-1)
        ax.set_aspect('equal', adjustable='box')
    
    def __call__(self):
        st.set_page_config(layout="wide")
        i = st.number_input("Environment", value=0, min_value=0, max_value=99)
        # ue_idx1 = st.number_input("#UE", value=0, min_value=0, max_value=29)
        env_info = self.get_env_info(i)
        
        n_cols = 3
        fig, ax = plt.subplots(nrows=2 * 30 // n_cols, ncols=n_cols, figsize=(24, 21 * 6), constrained_layout=True)
        # ax = [ax]
        
        for ue_idx1 in range(30):
            ax_idx = 2 * (ue_idx1 // n_cols), ue_idx1 % n_cols
            hist_ax_idx = 2 * (ue_idx1 // n_cols) + 1, ue_idx1 % n_cols
            heatmap_ax = ax[ax_idx]
            hist_ax = ax[hist_ax_idx]
            
            similarities = ValidationVisualize.get_similarities(
                ue_idx1, env_info.paths1, env_info.paths2, env_info.max_tau
            )
            ValidationVisualize.plot_similarities(fig, heatmap_ax, env_info, ue_idx1, similarities)
            
            pred_path = os.path.join(self.predictions_path, f"{env_info.env}_{ue_idx1}.npz")
            pred = list(np.load(pred_path, allow_pickle=True).values())[0]
            contours = ValidationVisualize.plot_prediction_contours(
                heatmap_ax, env_info.img_height, env_info.img_width, pred
            )
            
            scenario1_ue = env_info.ue_locs1[ue_idx1]
            ValidationVisualize.plot_histogram(
                hist_ax, env_info, pred, similarities, scenario1_ue, self.baseline_radius
            )
            
            ValidationVisualize.plot_legends(heatmap_ax, hist_ax, contours)
            ValidationVisualize.plot_details(heatmap_ax)
        
        st.pyplot(fig)
