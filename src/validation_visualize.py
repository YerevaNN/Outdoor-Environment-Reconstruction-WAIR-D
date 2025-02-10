from omegaconf import DictConfig

from src.visualizations import ValidationVisualize


def validation_visualize(config: DictConfig) -> None:
    validation_visualizer = ValidationVisualize(
        raw_data_path=config.raw_data_path,
        predictions_path=config.predictions_path,
        baseline_radius=config.baseline_radius
    )
    validation_visualizer()
