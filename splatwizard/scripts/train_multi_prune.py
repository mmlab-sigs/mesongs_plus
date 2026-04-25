"""
MesonGS Plus Multi-Prune Training Script

One-shot importance computation + multiple pruning rates.

Usage:
    python splatwizard/scripts/train_multi_prune.py \
        --source_path ... --yaml_path cfgs/mesongs/c1/train.yaml \
        --model mesongs_plus --optim mesongs_plus \
        --force_setup --iterations 1 --use_quat \
        --scene_imp train --images images \
        --codebook_size 4096 --num_bits 16 --raht True --use_indexed True \
        --sh_keep_threshold -1 --sh_keep_topk 1000000 \
        --init_checkpoint .../point_cloud.ply \
        --output_dir_template "outputs_jcge/mesongs_plus_{scene}_{config}_quat_train_nb{n_block}_bits{num_bits}_prune{percent}_cb{codebook_size}_topk{sh_keep_topk}_raht{raht}_use_indexed{use_indexed}"

The script reads pruning_rates from YAML, computes importance once,
then for each rate: deep-copies the model, prunes, builds octree, runs VQ,
optionally fine-tunes, and saves to rate-specific output directories.
"""

import copy
import pathlib
import sys
import yaml

from simple_parsing import ArgumentParser
from loguru import logger
import torch

from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.model_zoo import CONFIG_CACHE, init_model
from splatwizard.modules.dataclass import TrainContext
from splatwizard.modules.gaussian_model import GaussianModel
from splatwizard.pipeline.evaluation import evaluate
from splatwizard.scene import Scene
from splatwizard.scheduler import Scheduler
from splatwizard.utils.logging import setup_tensorboard
from splatwizard.utils.misc import safe_state


def _setup_output_dir(pp: PipelineParams, train_context: TrainContext):
    """Set up output directory, checkpoint dir, logger, tensorboard."""
    if pp.output_dir is None:
        return train_context

    train_context.output_dir = train_context.base_output_dir
    train_context.checkpoint_dir = train_context.output_dir / 'checkpoints'
    train_context.checkpoint_dir.mkdir(exist_ok=True)

    logger.add(train_context.output_dir / 'output.log')

    dataset = pathlib.Path(pp.source_path).name if pp.dataset is None else pp.dataset
    tb_writer = setup_tensorboard(train_context.output_dir)
    tb_writer.prefix = dataset
    train_context.tb_writer = tb_writer

    logger.info(f"Output dir: {train_context.output_dir.absolute()}")
    return train_context


def _run_finetune_loop(gs_model, pp, opt, scene, train_context):
    """Run the fine-tuning training loop (same as train_model but without pre-tasks)."""
    pre_scheduler = Scheduler()
    post_scheduler = Scheduler()
    # No pre-tasks registered: importance/prune/octree/vq already done

    gs_model.register_post_task(post_scheduler, pp, opt)

    first_iter = 1
    pre_scheduler.init(first_iter)
    post_scheduler.init(first_iter)

    bg_color = [1, 1, 1] if pp.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=pp.device)

    gs_model.train()
    cam_iterator = scene.getTrainCameras()
    task_cam_iterator = scene.get_task_train_cameras() if opt.camera_dependent_task else None

    for iteration in range(first_iter, opt.iterations + 1):
        pre_scheduler.exec_task(pp, opt, cam_iterator=task_cam_iterator)
        pre_scheduler.step()

        try:
            viewpoint_cam = next(cam_iterator)
        except StopIteration:
            viewpoint_cam = next(cam_iterator)

        bg = torch.rand((3,), device=pp.device) if opt.random_background else background
        render_result = gs_model.render(viewpoint_cam, bg, pp, opt, step=iteration)
        loss, loss_pack = gs_model.loss_func(viewpoint_cam, render_result, opt)
        if opt.iterations > 1:
            loss.backward()

        with torch.no_grad():
            if pp.eval_freq is not None and (iteration + 1) % pp.eval_freq == 0:
                eval_pack = evaluate(gs_model, pp, scene)
                gs_model.eval_report(eval_pack, iteration, train_context.tb_writer)

            post_scheduler.exec_task(pp, opt, render_result, task_cam_iterator)
            post_scheduler.step()

            if iteration < opt.iterations and opt.iterations > 1:
                gs_model.optimizer_step(render_result, opt, step=iteration)
                gs_model.optimizer.zero_grad(set_to_none=True)

            # Save checkpoints
            if train_context.checkpoint_dir is not None:
                if iteration in pp.checkpoint_iterations:
                    gs_model.save(train_context.checkpoint_dir, iteration, type_=pp.checkpoint_type)
                elif iteration == opt.iterations:
                    if isinstance(pp.final_checkpoint, str):
                        gs_model.save(train_context.checkpoint_dir, iteration, type_=pp.checkpoint_type)
                    else:
                        for type_ in pp.final_checkpoint:
                            gs_model.save(train_context.checkpoint_dir, iteration, type_=type_)

    return gs_model


def main():
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(PipelineParams, dest="pipeline")
    parser.add_arguments(CONFIG_CACHE[0], dest="model_group")
    # Extra argument for output dir template
    parser.add_argument(
        "--output_dir_template", type=str, default="",
        help="Python format string template for per-rate output dir. "
             "Available placeholders: {scene}, {config}, {n_block}, {num_bits}, "
             "{percent}, {codebook_size}, {sh_keep_topk}, {raht}, {use_indexed}"
    )
    # Optional CLI override: if given, takes precedence over YAML's pruning_rates
    parser.add_argument(
        "--pruning_rates_list", type=float, nargs='+', default=None,
        help="Pruning rates to train, e.g. --pruning_rates_list 0.2 0.4. "
             "If given, overrides the pruning_rates field in --yaml_path."
    )

    args = parser.parse_args(sys.argv[1:])

    mp = args.model_group.model
    pp: PipelineParams = args.pipeline
    op: OptimizationParams = args.model_group.optim
    output_dir_template = args.output_dir_template
    cli_pruning_rates = args.pruning_rates_list

    if pp.seed is not None:
        safe_state(pp.seed)

    # ---- Decide pruning_rates (CLI > YAML > --percent) ----
    pruning_rates = []
    if cli_pruning_rates:
        pruning_rates = list(cli_pruning_rates)
        logger.info(f"Using pruning_rates from CLI: {pruning_rates}")
    elif hasattr(mp, 'yaml_path') and mp.yaml_path:
        with open(mp.yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        pruning_rates = config_dict.get("pruning_rates", [])
        if pruning_rates:
            logger.info(f"Read pruning_rates from YAML: {pruning_rates}")

    if not pruning_rates:
        # Fallback: single rate from CLI --percent
        pruning_rates = [mp.percent]
        logger.info(f"No pruning_rates provided, using single rate from --percent: {pruning_rates}")

    # ---- Infer config name from yaml_path ----
    config_name = "c1"
    if hasattr(mp, 'yaml_path') and mp.yaml_path:
        parts = pathlib.Path(mp.yaml_path).parts
        if len(parts) >= 2:
            config_name = parts[-2]

    # ---- Build default output_dir_template ----
    if not output_dir_template:
        output_dir_template = (
            "outputs_jcge/mesongs_plus_{scene}_{config}_quat_train"
            "_nb{n_block}_bits{num_bits}_prune{percent}"
            "_cb{codebook_size}_topk{sh_keep_topk}"
            "_raht{raht}_use_indexed{use_indexed}"
        )

    # ---- Create scene & base model ----
    scene = Scene(pp, op)
    gs_model: GaussianModel = init_model("mesongs_plus", mp)
    if mp.require_cam_infos:
        gs_model.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent, scene.train_cameras[1.0])
    else:
        gs_model.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)

    # Load init_checkpoint (pretrained 3DGS .ply)
    if pp.init_checkpoint:
        logger.info(f"Loading init_checkpoint: {pp.init_checkpoint}")
        gs_model.load(pp.init_checkpoint, opt=op)

    if op.force_setup:
        gs_model.training_setup(op)

    gs_model.after_setup_hook(pp, op)

    # ---- Step 1: Compute importance ONCE ----
    logger.info("=" * 60)
    logger.info("Step 1: Computing importance (one-shot for all pruning rates)")
    logger.info("=" * 60)

    cam_iterator = scene.get_task_train_cameras()
    gs_model.compute_imp(cam_iterator, pp, op)

    logger.info(f"Importance computed: {gs_model.imp.shape[0]} points")
    logger.info(f"  imp range: [{gs_model.imp.min().item():.6f}, {gs_model.imp.max().item():.6f}]")

    # gs_model now holds the full (unpruned) point cloud + imp.
    # For each rate we deep-copy this base model, set percent, then prune.

    # ---- Step 2: For each pruning rate, fork the model ----
    for rate_idx, rate in enumerate(pruning_rates):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Step 2.{rate_idx + 1}: Processing pruning rate {rate} "
                    f"({rate_idx + 1}/{len(pruning_rates)})")
        logger.info("=" * 60)

        # Construct per-rate output dir
        fmt = {
            "scene": getattr(mp, 'scene_imp', ''),
            "config": config_name,
            "n_block": getattr(mp, 'n_block', 66),
            "num_bits": getattr(mp, 'num_bits', 8),
            "percent": rate,
            "codebook_size": getattr(mp, 'codebook_size', 2048),
            "sh_keep_topk": getattr(mp, 'sh_keep_topk', -1),
            "raht": getattr(mp, 'raht', True),
            "use_indexed": getattr(mp, 'use_indexed', True),
        }
        rate_output_dir = output_dir_template.format(**fmt)
        logger.info(f"  Output dir: {rate_output_dir}")

        # Deep-copy the base model (with imp, before pruning)
        rate_model = copy.deepcopy(gs_model)
        rate_model.percent = float(rate)

        # Prune
        logger.info(f"  Pruning with rate={rate} (keeping top {(1-rate)*100:.0f}%)")
        rate_model.prune_mask()
        logger.info(f"  After prune: {rate_model.xyz.shape[0]} points")

        # Octree coding
        # octree_coding / vq_fe 被 @task 装饰器包装，签名变成了
        #   wrapper(self, render_result, ppl, opt, step, cam_iterator)
        # 但原始函数只有 self，wrapper 内部不会使用这些参数，
        # 传 None/dummy 即可满足签名要求。
        logger.info(f"  Building octree...")
        rate_model.octree_coding(None, pp, op, 1, None)

        # VQ (if use_indexed)
        if rate_model.use_indexed:
            logger.info(f"  Running VQ quantization...")
            rate_model.vq_fe(None, pp, op, 1, None)

        # Setup training for fine-tuning
        rate_model.training_setup(op)

        # Set up per-rate output directory
        pp_rate = copy.deepcopy(pp)
        pp_rate.output_dir = rate_output_dir

        tc = TrainContext()
        tc.model = "mesongs_plus"
        tc.base_output_dir = pathlib.Path(rate_output_dir)
        tc.base_output_dir.mkdir(exist_ok=True, parents=True)
        _setup_output_dir(pp_rate, tc)

        # Run fine-tuning loop
        logger.info(f"  Fine-tuning ({op.iterations} iterations)...")
        _run_finetune_loop(rate_model, pp_rate, op, scene, tc)

        logger.info(f"  Done: rate={rate}, output={rate_output_dir}")

        # Free GPU memory
        del rate_model
        torch.cuda.empty_cache()

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"All {len(pruning_rates)} pruning rates completed!")
    logger.info("=" * 60)


if __name__ == '__main__':
    sys.exit(main())
