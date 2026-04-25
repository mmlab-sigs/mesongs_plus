from dataclasses import dataclass
from splatwizard.config import ModelParams, OptimizationParams


@dataclass
class ContextGSModelParams(ModelParams):
    sh_degree = 3
    feat_dim = 50
    n_offsets = 10
    voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
    update_depth = 3
    update_init_factor = 16
    update_hierachy_factor = 4

    use_feat_bank = False
    lod = 0

    hyper_divisor=4
    target_ratio=0.2

    n_features = 4
    level_num = 3
    disable_hyper = False
    lmbda_rec = 1
    lmbda = 0.001


@dataclass
class ContextGSOptimizationParams(OptimizationParams):
    require_pretrained = False
    iterations = 30_000
    position_lr_init = 0.0
    position_lr_final = 0.0
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000

    offset_lr_init = 0.01
    offset_lr_final = 0.0001
    offset_lr_delay_mult = 0.01
    offset_lr_max_steps = 30_000

    mask_lr_init = 0.01
    mask_lr_final = 0.0001
    mask_lr_delay_mult = 0.01
    mask_lr_max_steps = 30_000

    feature_lr = 0.0075
    hyper_latent_lr = 0.0075
    opacity_lr = 0.02
    scaling_lr = 0.007
    rotation_lr = 0.002

    mlp_opacity_lr_init = 0.002
    mlp_opacity_lr_final = 0.00002  
    mlp_opacity_lr_delay_mult = 0.01
    mlp_opacity_lr_max_steps = 30_000

    mlp_cov_lr_init = 0.004
    mlp_cov_lr_final = 0.004
    mlp_cov_lr_delay_mult = 0.01
    mlp_cov_lr_max_steps = 30_000

    mlp_color_lr_init = 0.008
    mlp_color_lr_final = 0.00005
    mlp_color_lr_delay_mult = 0.01
    mlp_color_lr_max_steps = 30_000

    mlp_featurebank_lr_init = 0.01
    mlp_featurebank_lr_final = 0.00001
    mlp_featurebank_lr_delay_mult = 0.01
    mlp_featurebank_lr_max_steps = 30_000

    latent_codec_lr_init = 0.005
    latent_codec_lr_final = 0.00001
    latent_codec_lr_delay_mult = 0.33
    latent_codec_lr_max_steps = 30_000

    mlp_grid_lr_init = 0.005
    mlp_grid_lr_final = 0.00001
    mlp_grid_lr_delay_mult = 0.01
    mlp_grid_lr_max_steps = 30_000

    mlp_deform_lr_init = 0.005
    mlp_deform_lr_final = 0.0005
    mlp_deform_lr_delay_mult = 0.01
    mlp_deform_lr_max_steps = 30_000

    percent_dense = 0.01
    lambda_dssim = 0.2

    # for anchor densification
    start_stat = 500
    update_from = 1500
    update_interval = 100
    update_until = 15_000

    min_opacity = 0.005  # 0.2
    success_threshold = 0.8
    densify_grad_threshold = 0.0002
