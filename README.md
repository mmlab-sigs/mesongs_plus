# MesonGS++

A clean, reproducible implementation of **MesonGS++**, our pipeline for 3D
Gaussian Splatting compression. Built on top of the
[SplatWizard](https://github.com/) framework.

Supported benchmarks (released scripts cover **13 scenes**):

- **Mip-NeRF 360** — `bicycle`, `bonsai`, `counter`, `garden`, `kitchen`, `room`, `stump`, `flowers`, `treehill`
- **Tanks and Temples** — `train`, `truck`
- **Deep Blending** — `drjohnson`, `playroom`


We release the full set of compressed bit-streams and
per-point `results.json` for all scenes (Mip-NeRF 360 + Tanks and Temples +
Deep Blending) as a single archive:

- **Google Drive**: [https://drive.google.com/file/d/1xhajsEVy5bnQayTtmM1apTMSWzlic572/view?usp=sharing](https://drive.google.com/file/d/1xhajsEVy5bnQayTtmM1apTMSWzlic572/view?usp=sharing)

## Install

```bash
pip install torch==2.4.0+cu121 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install -r requirements.txt
pip install -e .
```

### External dependency: MPEG G-PCC codec (tmc3)

RD evaluation uses `tmc3` from
[MPEG PCC TMC13](https://github.com/MPEGGroup/mpeg-pcc-tmc13). Build it and
expose the binary path:

```bash
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git
cd mpeg-pcc-tmc13 && mkdir build && cd build && cmake .. && make -j
export TMC3_PATH=$(pwd)/tmc3/tmc3
```

## Repository layout

```
mesongs++/
├── cfgs/
│   ├── decoder.cfg                     # GPCC decoder config
│   ├── lossless_encoder.cfg            # GPCC lossless encoder config
│   └── mesongs/c1/                     # 13 scene YAMLs (360 + tandt + db)
├── scripts/                            # shell entry points
│   ├── eval_mesongs_plus_360.sh        # Prune + RD eval: Mip-NeRF 360 (9 scenes)
│   ├── eval_mesongs_plus_tandt.sh      # Prune + RD eval: Tanks and Temples
│   ├── eval_mesongs_plus_db.sh         # Prune + RD eval: Deep Blending
│   └── compress_single_scene.sh        # Single scene + custom rates + single size_limit
└── splatwizard/
    ├── scripts/                        # python CLI entry points
    │   ├── train.py                    # standard single-rate training
    │   ├── train_multi_prune.py        # one-shot importance + multi-rate pruning (ours)
    │   ├── eval.py                     # single-point evaluation
    │   └── eval_rd_curve.py            # RD-curve evaluation
    ├── pipeline/                       # train_model / eval_model / rd_curve / evaluation
    ├── model_zoo/
    │   ├── mesongs/                    # MesonGS baseline
    │   ├── mesongs_plus/               # MesonGS++ (our method)
    │   └── {gs,hac,cat_3dgs,compactgs,...}  # other baselines shipped with SplatWizard
    ├── rasterizer/                     # python-level rasterizer wrappers
    ├── _cmod/                          # native CUDA extensions (built by setup.py)
    ├── modules/, compression/, metrics/, scene/, utils/, ...
    └── config.py, scheduler.py, ...
```

> The repository keeps all SplatWizard baselines so that users can reproduce
> comparison experiments from our paper, but **only MesonGS++ shell scripts
> under `scripts/` are officially released**. Other baselines can be run
> directly via `splatwizard/scripts/train.py` / `eval.py`.


## Usage

### 1. Full dataset: Prune + RD-curve evaluation

```bash
bash scripts/eval_mesongs_plus_360.sh     # Mip-NeRF 360
bash scripts/eval_mesongs_plus_tandt.sh   # Tanks and Temples
bash scripts/eval_mesongs_plus_db.sh      # Deep Blending
```

Each scene YAML (e.g. `cfgs/mesongs/c1/bicycle.yaml`) specifies a
`pruning_rates` list (default `[0.2, 0.4]`) and a `rd_curve_size_limits` list
(RD operating points in MB).  Each dataset script runs two stages:

1. **Prune**: for every scene and every rate in `pruning_rates`, MesonGS++
   computes point importance *once* and forks the model for each rate.
   Checkpoints are saved to
   ```
   outputs_jcge/mesongs_plus_{scene}_c1_quat_train_nb{nb}_bits{b}_prune{rate}_cb{cb}_topk{topk}_raht{raht}_use_indexed{idx}/checkpoints/ckpt1.pth
   ```
2. **RD-curve eval**: for every RD point in `rd_curve_size_limits`, try all
   per-rate checkpoints and keep the one with the highest PSNR.

Edit the `SCENES=(...)` array and the hyper-parameters at the top of each
script to restrict the run (e.g. to a single scene).

### 2. Single scene with custom pruning rates and a single size target

For quick experiments on any **user-provided** scene with your own
`pruning_rates` list and a specific target bit-stream size, use
`compress_single_scene.sh`. The script only requires two paths:

- `SOURCE_PATH`     — COLMAP/NeRF-Synthetic scene directory (contains `images/` + `sparse/` or equivalent)
- `INIT_CHECKPOINT` — a pretrained 3DGS `point_cloud.ply` (from the official 3DGS training pipeline)

```bash
# minimal: use built-in defaults (counter scene, rates=[0.2, 0.4], size=20 MB)
bash scripts/compress_single_scene.sh
```

Pipeline:

1. Auto-generate a minimal YAML from the given hyper-parameters
   (`$OUTPUT_ROOT/<tag>_config.yaml`).
2. Run `splatwizard/scripts/train_multi_prune.py` to produce one compressed
   checkpoint per rate in `PRUNING_RATES`.
3. Run `splatwizard/scripts/eval.py` once per checkpoint with the specified
   `--size_limit_mb`, then print a comparison table of PSNR / SSIM / LPIPS so
   you can pick the best rate manually.

Useful environment variables (all optional):

| Variable          | Default                        | Meaning                                  |
| ----------------- | ------------------------------ | ---------------------------------------- |
| `PRUNING_RATES` | `"0.2 0.4"`                  | Space-separated pruning rates to try     |
| `SIZE_LIMIT_MB` | `20`                         | Target bit-stream size (MB)              |
| `OUTPUT_ROOT`   | `outputs_single`             | Root directory for all artifacts         |
| `TAG`           | `$(basename "$SOURCE_PATH")` | Used as prefix for per-rate output dirs  |
| `CUDA_DEVICE`   | `0`                          | GPU id                                   |
| `IMAGES`        | `images`                     | COLMAP images subdir                     |
| `SKIP_PRUNE`    | `0`                          | `=1` to skip the pruning stage         |
| `OCTREE_DEPTH`  | `19`                         | GPCC octree depth                        |
| `N_BLOCK`       | `80`                         | RAHT block count                         |
| `CODEBOOK_SIZE` | `4096`                       | VQ codebook size                         |
| `NUM_BITS`      | `16`                         | Quantizer bit width                      |
| `RAHT`          | `True`                       | Use RAHT transform                       |
| `USE_INDEXED`   | `True`                       | Use indexed rasterizer / SH quantization |

## Scene YAML format

For full-dataset scripts, every scene needs a YAML under
`cfgs/mesongs/c1/<scene>.yaml`:

```yaml
# cfgs/mesongs/c1/bicycle.yaml
n_block: 80
cb: 2048
depth: 19
prune: 0.4
finetune_lr_scale: 0.1

# MesonGS++ specific fields
pruning_rates: [0.2, 0.4]               # rates trained in Step 1
rd_curve_size_limits: [109.2, 95.7, 83.8, 71.5, 62.2]   # MB, used in Step 2
```

For single-scene runs (`compress_single_scene.sh`) the YAML is generated
automatically; no manual config file is needed.

## License

See [LICENSE.md](LICENSE.md).
