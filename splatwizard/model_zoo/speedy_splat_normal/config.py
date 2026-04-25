from dataclasses import dataclass
from splatwizard.config import ModelParams, OptimizationParams


@dataclass
class SpeedySplatNormalModelParams(ModelParams):
    sh_degree: int = 3


@dataclass
class SpeedySplatNormalOptimizationParams(OptimizationParams):
    require_pretrained: bool = False
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01

    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    random_background: bool = False
    normal_regularity_from_iter: int = 7000
    normal_regularity_until_iter: int = 12000

    # pruning parameters
    prune_from_iter: int = 6000
    prune_until_iter: int = 30_000
    prune_interval: int = 3000
    densify_prune_ratio: float = 0.80
    after_densify_prune_ratio: float = 0.30

    densify_scale_factor: float = 1.0
    # normal_regularity_from_iter = 500
    normal_regularity_param: float = 0.05
    normal_close_thresh: float = 1.0
    neighbor_reset_interval: int = 500
    normal_dilation: int = 2
    depth_grad_thresh: float = 0.05
    depth_grad_mask_dilation = 1
    contribution_prune_from_iter = 1000
    contribution_prune_interval = 500
    contribution_prune_ratio: float = 0.1
    knn_to_track: int = 16

