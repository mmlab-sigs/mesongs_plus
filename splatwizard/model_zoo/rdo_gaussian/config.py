from dataclasses import dataclass
from splatwizard.config import ModelParams, OptimizationParams


@dataclass
class RDOGaussianModelParams(ModelParams):
    sh_degree: int = 3
    vq_scale_cbsize: int = 8192
    vq_rot_cbsize: int = 8192
    vq_dc_cbsize: int = 8192
    vq_sh1_cbsize: int = 4096
    vq_sh2_cbsize: int = 4096
    vq_sh3_cbsize: int = 4096
    vq_patch_size: int = 16384


@dataclass
class RDOGaussianOptimizationParams(OptimizationParams):
    require_pretrained = False
    iterations: int = 30_000
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01

    lambda_dssim: float = 0.2

    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002
    random_background = False

    # Gaussian prune cfgs
    gs_prune_start_iter: int = 15_001
    gs_mask_lr: float = 0.01
    gs_mask_lambda: float = 0.0005
    # Adaptive SHs prune cfgs
    sh_prune_start_iter: int = 15_001
    sh_mask_lr: float = 0.05
    sh_mask_lambda: float = 0.005
    # VQ cfgs
    vq_start_iter: int = 15_001
    rate_constrain_iter: int = 20_001
    reactivate_codeword_period: int = 1000
    vq_cb_lr: float = 0.0002
    vq_logits_lr: float = 0.002
    vq_scale_lmbda: int = 32768
    vq_rot_lmbda: int = 256
    vq_dc_lmbda: int = 256
    vq_sh1_lmbda: int = 256
    vq_sh2_lmbda: int = 256
    vq_sh3_lmbda: int = 256


