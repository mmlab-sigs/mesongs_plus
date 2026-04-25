from __future__ import annotations

import copy
import json
import multiprocessing as py_mp
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from loguru import logger

from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.dataclass import TrainContext
from splatwizard.pipeline.eval_model import eval_model


def _eval_model_worker(
    ppl: PipelineParams,
    model_params: Any,
    opt: OptimizationParams,
    model_name: str,
    run_dir: str,
) -> None:
    """
    Run one eval in a fresh process to avoid CUDA context / extension state leaking across
    multiple RD-curve points (a common cause of intermittent 'illegal memory access').
    """
    # Child process imports must be inside the worker for spawn mode robustness.
    import pathlib

    tc = TrainContext()
    tc.model = model_name
    tc.base_output_dir = pathlib.Path(run_dir)
    tc.base_output_dir.mkdir(parents=True, exist_ok=True)
    tc.output_dir = tc.base_output_dir
    tc.tb_writer = None

    # Set up render_result_dir if save_rendered_image is enabled,
    # otherwise save_image will fail with NoneType / str error.
    if ppl.save_rendered_image:
        tc.render_result_dir = tc.output_dir / 'render_results'
        tc.render_result_dir.mkdir(parents=True, exist_ok=True)

    eval_model(ppl, model_params, opt, tc)

def _read_result(run_dir: Path) -> Dict[str, Any]:
    """Read results.json from a run directory and extract psnr / total_bytes."""
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {"ok": False, "psnr": None, "total_bytes": None, "size_mb": None}

    with open(results_path, "r") as f:
        payload = json.load(f)

    psnr = payload.get("psnr") or payload.get("psnr_val") or payload.get("PSNR")
    total_bytes = payload.get("total_bytes") or payload.get("bytes") or payload.get("bitstream_bytes")

    size_mb = None
    if total_bytes is not None:
        try:
            size_mb = float(total_bytes) / (1024.0 * 1024.0)
        except Exception:
            size_mb = None

    return {"ok": True, "psnr": psnr, "total_bytes": total_bytes, "size_mb": size_mb}


def _run_single_point(
    ctx,
    pp: PipelineParams,
    mp,
    op: OptimizationParams,
    model_name: str,
    target_limit: float,
    target_pr: float,
    run_dir: Path,
    checkpoint_override: str = None,
    percent_override: float = None,
) -> Dict[str, Any]:
    """
    Run a single RD evaluation point in a spawned subprocess.
    Returns a result dict with psnr, size_mb, etc.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    ppl_i = copy.deepcopy(pp)
    mp_i = copy.deepcopy(mp)
    mp_i.size_limit_mb = float(target_limit)
    mp_i.pruning_rate = float(target_pr)

    # Override checkpoint and percent if provided (multi-checkpoint mode)
    if checkpoint_override is not None:
        ppl_i.checkpoint = checkpoint_override
    if percent_override is not None:
        mp_i.percent = float(percent_override)

    proc = ctx.Process(
        target=_eval_model_worker,
        args=(ppl_i, mp_i, op, model_name, str(run_dir)),
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        logger.error(
            f"RD point failed (limit={target_limit:.3f}MB, prune={target_pr}"
            f"{f', ckpt_percent={percent_override}' if percent_override is not None else ''}) "
            f"with exitcode={proc.exitcode}. See logs under {run_dir}."
        )
        return {"ok": False, "psnr": None, "total_bytes": None, "size_mb": None, "run_dir": str(run_dir)}

    res = _read_result(run_dir)
    res["run_dir"] = str(run_dir)
    return res


def _build_checkpoint_path(mp, percent: float) -> str:
    """
    Auto-construct checkpoint path from model parameters and a given training percent.

    Uses mp.checkpoint_template if set; otherwise falls back to the standard naming convention.
    Returns the first existing path among candidates, or the primary path if none exist.
    """
    # Determine template
    if hasattr(mp, 'checkpoint_template') and mp.checkpoint_template:
        template = mp.checkpoint_template
    else:
        template = (
            "mesongs_plus_{scene}_{config}_quat_train_nb{n_block}_bits{num_bits}"
            "_prune{percent}_cb{codebook_size}_topk{sh_keep_topk}"
            "_raht{raht}_use_indexed{use_indexed}/checkpoints/ckpt1.pth"
        )

    # Collect parameter values for formatting
    fmt = {
        "scene": getattr(mp, 'scene_imp', ''),
        "config": _infer_config_name(mp),
        "n_block": getattr(mp, 'n_block', 66),
        "num_bits": getattr(mp, 'num_bits', 8),
        "percent": percent,
        "codebook_size": getattr(mp, 'codebook_size', 2048),
        "sh_keep_topk": getattr(mp, 'sh_keep_topk', -1),
        "raht": getattr(mp, 'raht', True),
        "use_indexed": getattr(mp, 'use_indexed', True),
    }

    rel_path = template.format(**fmt)

    # Candidate base directories (try in order)
    candidate_bases = [
        Path("outputs_autotune"),
        Path("outputs_jcge"),
        Path("/home/gejunchen/Work/2026-1/Projects/compressgs/outputs_jcge"),
    ]

    for base in candidate_bases:
        full = base / rel_path
        if full.exists():
            logger.info(f"  Found checkpoint: {full}")
            return str(full)

    # Nothing found — return the first candidate so the error message is informative
    primary = candidate_bases[0] / rel_path
    logger.warning(f"  Checkpoint NOT found in any candidate dir, using: {primary}")
    return str(primary)


def _infer_config_name(mp) -> str:
    """Infer the config short name (e.g. 'c1') from mp.yaml_path."""
    yaml_path = getattr(mp, 'yaml_path', '')
    if yaml_path:
        # e.g. "cfgs/mesongs/c1/train.yaml" -> "c1"
        parts = Path(yaml_path).parts
        # The config name is typically the parent directory of the scene yaml
        if len(parts) >= 2:
            return parts[-2]
    return "c1"


def run_rd_curve_pipeline(pp: PipelineParams, mp, op: OptimizationParams, train_context: TrainContext):
    """
    Executes the Rate-Distortion curve evaluation pipeline.

    It iterates through a range of size limits (based on mp.size_limit_mb or mp.rd_curve_size_limits),
    optionally combined with pruning rates (mp.rd_curve_pruning_rates),
    performs encode-decode-eval for each, and plots the Size vs PSNR curve.

    Multi-checkpoint mode (pruning_rates):
        When mp.pruning_rates is non-empty (e.g. [0.2, 0.4]), for each RD point
        (size_limit × eval_pruning_rate), ALL training-time checkpoints (one per
        pruning rate) are evaluated. The result with the **highest PSNR** is selected
        as the final result for that point. Checkpoint paths are auto-constructed
        from model parameters + naming convention.
    """
    # Check if rd_curve_size_limits is provided (via config or CLI)
    size_limits = None
    if hasattr(mp, 'rd_curve_size_limits') and mp.rd_curve_size_limits:
        size_limits = [float(x) for x in mp.rd_curve_size_limits]
        logger.info(f"Using custom size_limits from config/CLI: {size_limits}")
    elif mp.size_limit_mb <= 0:
        logger.warning(
            "mp.size_limit_mb is not positive and no rd_curve_size_limits provided. "
            "RD Curve cannot be generated properly. "
            "Please set a valid limit in config or CLI."
        )
        return

    base_limit = float(mp.size_limit_mb)

    # Use custom size_limits if provided, otherwise use default ratios
    if size_limits is None:
        ratios: List[float] = [0.65, 0.7, 0.83, 0.95, 1.0]
        size_limits = [base_limit * r for r in ratios]
        logger.info(f"Using default ratios with base limit {base_limit} MB")
    else:
        logger.info(f"Using {len(size_limits)} custom size limits")

    # Check if rd_curve_pruning_rates is provided (eval-time re-pruning sweep)
    eval_pruning_rates: List[float] = []
    if hasattr(mp, 'rd_curve_pruning_rates') and mp.rd_curve_pruning_rates:
        eval_pruning_rates = [float(x) for x in mp.rd_curve_pruning_rates]
        logger.info(f"Using eval pruning_rates sweep: {eval_pruning_rates}")
    
    # Build list of (size_limit, pruning_rate) tuples to evaluate
    eval_points: List[Tuple[float, float]] = []
    if eval_pruning_rates:
        # Sweep over (size_limit, pruning_rate) pairs
        for sl in size_limits:
            for pr in eval_pruning_rates:
                eval_points.append((sl, pr))
    else:
        # Only sweep size_limits (pruning_rate uses whatever is set in mp or -1.0)
        current_pr = getattr(mp, 'pruning_rate', -1.0)
        for sl in size_limits:
            eval_points.append((sl, current_pr))

    # ---- Multi-checkpoint mode via pruning_rates ----
    # Build checkpoint_configs from mp.pruning_rates (training-time pruning rates).
    # Each entry: {"checkpoint": auto-derived path, "percent": rate}
    checkpoint_configs: List[Dict] = []
    if hasattr(mp, 'pruning_rates') and mp.pruning_rates:
        logger.info(f"Building checkpoint configs from pruning_rates: {mp.pruning_rates}")
        for rate in mp.pruning_rates:
            ckpt_path = _build_checkpoint_path(mp, rate)
            checkpoint_configs.append({
                "checkpoint": ckpt_path,
                "percent": float(rate),
            })
        logger.info(f"Multi-checkpoint mode ENABLED: {len(checkpoint_configs)} checkpoint configs")
        for ci, cfg in enumerate(checkpoint_configs):
            logger.info(f"  [{ci}] percent={cfg['percent']}, checkpoint={cfg['checkpoint']}")
    
    multi_ckpt_mode = len(checkpoint_configs) > 0

    n_total = len(eval_points) * max(len(checkpoint_configs), 1)
    logger.info(f"Starting RD Curve evaluation: {len(eval_points)} RD points"
                f"{f' × {len(checkpoint_configs)} checkpoints = {n_total} runs' if multi_ckpt_mode else ''}")
    logger.info(f"  size_limits: {size_limits}")
    if eval_pruning_rates:
        logger.info(f"  eval_pruning_rates: {eval_pruning_rates}")

    save_dir = Path(train_context.output_dir) / "rd_curve"
    save_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    all_candidate_results: List[Dict[str, Any]] = []  # all individual runs (for debugging)

    # Important: run each point in a fresh process to avoid CUDA extension state leaking
    # across iterations (this often manifests as intermittent 'illegal memory access').
    ctx = py_mp.get_context("spawn")

    for idx, (target_limit, target_pr) in enumerate(eval_points):

        if multi_ckpt_mode:
            # ======== Multi-checkpoint mode: try all checkpoints, pick best PSNR ========
            logger.info(f"=== RD Point {idx+1}/{len(eval_points)}, "
                        f"target_limit={target_limit:.3f} MB, pruning_rate={target_pr} "
                        f"(evaluating {len(checkpoint_configs)} checkpoints) ===")

            candidates: List[Dict[str, Any]] = []
            for ci, ckpt_cfg in enumerate(checkpoint_configs):
                ckpt_path = ckpt_cfg["checkpoint"]
                ckpt_percent = float(ckpt_cfg["percent"])

                if target_pr >= 0:
                    run_dir = save_dir / (
                        f"point_{idx:02d}_limit_{target_limit:.3f}MB_prune_{target_pr:.3f}"
                        f"_ckpt_percent{ckpt_percent:.2f}"
                    )
                else:
                    run_dir = save_dir / (
                        f"point_{idx:02d}_limit_{target_limit:.3f}MB"
                        f"_ckpt_percent{ckpt_percent:.2f}"
                    )

                logger.info(f"  [{ci+1}/{len(checkpoint_configs)}] "
                            f"checkpoint percent={ckpt_percent}, path={ckpt_path}")

                res = _run_single_point(
                    ctx, pp, mp, op, train_context.model,
                    target_limit, target_pr, run_dir,
                    checkpoint_override=ckpt_path,
                    percent_override=ckpt_percent,
                )
                res["ckpt_percent"] = ckpt_percent
                res["ckpt_path"] = ckpt_path
                candidates.append(res)

                if res["ok"]:
                    logger.info(f"    -> psnr={res['psnr']}, size_mb={res['size_mb']}")
                else:
                    logger.warning(f"    -> FAILED")

                # Also record in all_candidate_results for full debug output
                all_candidate_results.append({
                    "rd_point_index": idx,
                    "target_limit_mb": target_limit,
                    "pruning_rate": target_pr,
                    **res,
                })

            # Pick the candidate with the highest PSNR
            valid_candidates = [c for c in candidates if c["ok"] and c.get("psnr") is not None]
            if valid_candidates:
                best = max(valid_candidates, key=lambda c: float(c["psnr"]))
                logger.info(
                    f"  >> Best for point {idx}: psnr={best['psnr']}, size_mb={best['size_mb']}, "
                    f"from checkpoint percent={best['ckpt_percent']}"
                )
                results.append({
                    "index": idx,
                    "target_limit_mb": target_limit,
                    "pruning_rate": target_pr,
                    "ok": True,
                    "psnr": best["psnr"],
                    "total_bytes": best["total_bytes"],
                    "size_mb": best["size_mb"],
                    "run_dir": best["run_dir"],
                    "selected_ckpt_percent": best["ckpt_percent"],
                    "selected_ckpt_path": best["ckpt_path"],
                    "all_candidates": [
                        {"ckpt_percent": c.get("ckpt_percent"), "psnr": c.get("psnr"),
                         "size_mb": c.get("size_mb"), "ok": c.get("ok")}
                        for c in candidates
                    ],
                })
            else:
                logger.error(f"  >> All candidates failed for point {idx}")
                results.append({
                    "index": idx,
                    "target_limit_mb": target_limit,
                    "pruning_rate": target_pr,
                    "ok": False,
                    "psnr": None,
                    "total_bytes": None,
                    "size_mb": None,
                    "run_dir": None,
                    "all_candidates": [
                        {"ckpt_percent": c.get("ckpt_percent"), "psnr": c.get("psnr"),
                         "size_mb": c.get("size_mb"), "ok": c.get("ok")}
                        for c in candidates
                    ],
                })
        else:
            # ======== Single-checkpoint mode (original behavior) ========
            if target_pr >= 0:
                run_dir = save_dir / f"point_{idx:02d}_limit_{target_limit:.3f}MB_prune_{target_pr:.3f}"
            else:
                run_dir = save_dir / f"point_{idx:02d}_limit_{target_limit:.3f}MB"

            logger.info(f"--- RD Point {idx+1}/{len(eval_points)}, "
                        f"target_limit={target_limit:.3f} MB, pruning_rate={target_pr} ---")

            res = _run_single_point(
                ctx, pp, mp, op, train_context.model,
                target_limit, target_pr, run_dir,
            )

            if res["ok"]:
                logger.info(f"RD point done: size_mb={res['size_mb']}, psnr={res['psnr']}, pruning_rate={target_pr}")

            results.append({
                "index": idx,
                "target_limit_mb": target_limit,
                "pruning_rate": target_pr,
                **{k: res[k] for k in ("ok", "psnr", "total_bytes", "size_mb", "run_dir")},
            })

    # Save raw data
    results_json_path = save_dir / "rd_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"RD raw results saved to {results_json_path}")

    # Save all candidate results if multi-checkpoint mode was used (for debugging)
    if multi_ckpt_mode and all_candidate_results:
        all_results_path = save_dir / "rd_all_candidates.json"
        with open(all_results_path, "w") as f:
            json.dump(all_candidate_results, f, indent=2)
        logger.info(f"All candidate results saved to {all_results_path}")

    # Plot (only successful points with numeric values)
    xs: List[float] = []
    ys: List[float] = []
    for r in results:
        if not r.get("ok"):
            continue
        if r.get("size_mb") is None or r.get("psnr") is None:
            continue
        try:
            xs.append(float(r["size_mb"]))
            ys.append(float(r["psnr"]))
        except Exception:
            continue

    if len(xs) >= 1:
        plt.figure()
        plt.plot(xs, ys, marker="o")
        n_points = len(eval_points)
        title = f"Rate-Distortion Curve ({n_points} points)"
        if multi_ckpt_mode:
            title += f" [best of {len(checkpoint_configs)} ckpts]"
        plt.title(title)
        plt.xlabel("Size (MB)")
        plt.ylabel("PSNR")
        plt.grid(True)

        plot_path = save_dir / "rd_curve.png"
        plt.savefig(str(plot_path))
        logger.info(f"RD Curve saved to {plot_path}")
    else:
        logger.warning("No valid RD points to plot (all failed or missing psnr/size).")


