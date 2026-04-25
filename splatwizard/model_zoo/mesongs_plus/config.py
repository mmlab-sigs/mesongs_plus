from dataclasses import dataclass, field
from enum import Enum
from typing import List

from splatwizard.config import ModelParams, OptimizationParams, PipelineParams


class Stage(Enum):
    TRAIN = 1
    PRUNE = 2
    DISTILL = 3
    ENCODE = 4


@dataclass
class MesonGSPlusModelParams(ModelParams):
    sh_degree: int = 3
    save_imp: bool = False
    depth_count: bool = False
    save_mode: str = 'euler'
    not_update_rot: bool = False
    skip_quant_rot: bool = False
    hyper_config: str = "universal"
    save_ft_type: str = ""
    n_block: int = 66
    eval: bool = False
    codeft: bool = False
    no_simulate: bool = False
    oct_merge: str = "mean"
    codebook_size: int = 2048
    batch_size: int = 262144
    steps: int = 2000
    raht: bool = True
    percent: float = 0.66
    meson_count: bool = True
    f_count: bool = False
    debug: bool = False
    lseg: int = -1
    csv_path: str = ''
    depth: int = 12
    num_bits: int = 8
    clamp_color: bool = True
    per_channel_quant: bool = False
    per_block_quant: bool = True
    use_indexed: bool = True
    scene_imp: str = "" # scene_name
    yaml_path: str = "" # REQUIRE, config path
    finetune_lr_scale: float = 1.0
    use_quat: bool = True
    size_limit_mb: float = 100
    sh_keep_threshold: float = -1.0  # raw importance threshold for keeping full SH (e.g. 0.5, 1.0, 5.0), -1 to disable
    sh_keep_topk: int = -1  # keep top-K most important points with full SH (e.g. 10000, 50000), -1 to disable. If set, overrides sh_keep_threshold
    # Golden section search for optimal new_num_keep (replaces ternary search, ~2x faster)
    enable_golden_search: bool = False  # enable golden section search for new_num_keep
    golden_search_interval: int = 1000  # stop golden section search when interval < this value
    # Binary search for optimal new_num_keep (preferred over ternary search)
    enable_binary_search: bool = False  # enable binary search for new_num_keep using search_qbits objective value
    binary_search_interval: int = 5000  # stop binary search when interval < this value
    obj_threshold_ratio: float = 2.0  # objective value threshold as a multiplier of baseline obj (num_keep=0). e.g. 2.0 means allowing obj to double
    # RD curve specific: list of size limits in MB. If empty, will use ratios * size_limit_mb
    rd_curve_size_limits: List[float] = field(default_factory=list)  # e.g., [30, 40, 50, 60, 70, 80, 90, 100]
    # Eval-time re-pruning rate: prune this fraction of points (by importance) before encoding.
    # -1.0 means disabled (no re-pruning, use checkpoint as-is).
    # Valid range: [0.0, 1.0). E.g., 0.3 means prune 30% of least important points.
    pruning_rate: float = -1.0
    cb_quant_bits: int = 8  # quantization bits for kept points (codebook), default 8
    # Independent n_block for NTK (VQ index) Laplace encoding.
    # -1 means use the same n_block as RAHT. VQ indices have different statistics
    # (uniform-ish integers 0..4095 vs real-valued RAHT AC coefficients), so
    # an independent setting may improve coding efficiency.
    ntk_n_block: int = -1
    # Independent n_block for codebook (kept SH) Laplace encoding.
    # -1 means use the same n_block as RAHT.
    cb_n_block: int = -1
    # Eval-time num_keep: directly specify the number of kept points.
    # -1 means use default logic (_adjust_num_keep_for_size_limit).
    # >= 0 means use this exact value (bypass golden/binary search and default logic).
    num_keep: int = -1
    # RD curve specific: list of pruning rates to sweep. If empty, uses single pruning_rate.
    rd_curve_pruning_rates: List[float] = field(default_factory=list)  # e.g., [0.0, 0.1, 0.2, 0.3, 0.4]
    # Training & eval: list of training-time pruning rates (percent values).
    # - train: the train script reads this list and runs training for each rate.
    # - eval RD: the pipeline iterates all rates, auto-constructs the checkpoint path
    #   for each, evaluates them, and picks the best PSNR per RD point.
    # If empty, falls back to single --checkpoint / --percent from CLI.
    pruning_rates: List[float] = field(default_factory=list)  # e.g., [0.2, 0.4]
    # Checkpoint path template for auto-constructing checkpoint paths from pruning_rates.
    # Available placeholders: {scene}, {config}, {n_block}, {num_bits}, {percent},
    #   {codebook_size}, {sh_keep_topk}, {raht}, {use_indexed}
    # The template should include the full path up to and including ckpt1.pth.
    # Default uses the standard naming convention.
    checkpoint_template: str = ""
    # Improvement experiments: channel importance weight for ILP objective
    channel_importance_weight: bool = False
    # Improvement experiments: percentile quantization (clip outliers before quant)
    percentile_quant: bool = False
    # Improvement experiments: auto entropy model selection (Gaussian vs Laplace)
    auto_entropy_model: bool = False


@dataclass
class MesonGSPlusOptimizationParams(OptimizationParams):
    require_pretrained = False
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 10_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.001
    rotation_lr = 0.001
    percent_dense = 0.01
    
    lambda_dssim = 0.2
    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002

    # current_stage: Stage = None
