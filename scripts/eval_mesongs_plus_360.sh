#!/bin/bash
# ============================================================================
# MesonGS Plus 全流程（Prune + RD 曲线评估） - Mip-NeRF 360
#
# 1. Prune：对每个场景在 YAML 中声明的所有 pruning_rates 上运行剪枝
#    （一次 compute_imp + 多次 prune），checkpoint 保存到
#    outputs_jcge/.../checkpoints/ckpt1.pth
# 2. RD Eval：对每个场景跑多 checkpoint RD 曲线评估，pipeline
#    自动读取 YAML 的 pruning_rates 找对应 checkpoint，并在每个
#    RD 点自动选 PSNR 更高者。
# ============================================================================

set -e

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate compressgs

CONFIG=c1
DS=360_v2

# 可调超参数（prune + eval 共用）
N_BLOCK=80
CODEBOOK_SIZE=4096
NUM_BITS=16
RAHT=True
USE_INDEXED=True
SH_KEEP_THRESHOLD=-1
SH_KEEP_TOPK=1000000

SCENES=('bicycle' 'bonsai' 'counter' 'garden' 'kitchen' 'room' 'stump' 'flowers' 'treehill')
# 示例：只跑单个场景
# SCENES=('counter')

# 数据与外部工具路径
DATA_ROOT=${DATA_ROOT:-/mnt/storage/users/szxie_data/nerf_data}
INIT_CKPT_ROOT=${INIT_CKPT_ROOT:-/mnt/storage/users/szxie_data/MesonGS/outputs}
export TMC3_PATH=${TMC3_PATH:-/home/gejunchen/Work/2026-1/Projects/compressgs_autotune/mpeg-pcc-tmc13/build/tmc3/tmc3}

CUDA_DEVICE=${CUDA_DEVICE:-0}

# ============================================================================
# Step 1: Pruning (multi-prune)
# ============================================================================
for SCENE in ${SCENES[@]}
do
    YAML_PATH="cfgs/mesongs/$CONFIG/$SCENE.yaml"
    SOURCE_PATH="$DATA_ROOT/$DS/$SCENE"
    INIT_CHECKPOINT="$INIT_CKPT_ROOT/$DS/$SCENE/baseline/3dgs/point_cloud/iteration_30000/point_cloud.ply"

    echo "=========================================="
    echo "[Step 1/2] Pruning $SCENE (multi pruning_rates from YAML)"
    echo "  YAML: $YAML_PATH"
    echo "=========================================="

    PYTHONPATH=$PYTHONPATH:$(pwd) CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python splatwizard/scripts/train_multi_prune.py \
        --source_path "$SOURCE_PATH" \
        --yaml_path "$YAML_PATH" \
        --model mesongs_plus \
        --optim mesongs_plus \
        --force_setup \
        --iterations 1 \
        --checkpoint_iterations 199 399 599 1999 2999 3999 4999 5999 6999 7999 \
        --checkpoint_type pth \
        --final_checkpoint pth \
        --scene_imp "$SCENE" \
        --images images \
        --use_quat \
        --eval_freq 1 \
        --n_block $N_BLOCK \
        --codebook_size $CODEBOOK_SIZE \
        --num_bits $NUM_BITS \
        --raht $RAHT \
        --use_indexed $USE_INDEXED \
        --sh_keep_threshold $SH_KEEP_THRESHOLD \
        --sh_keep_topk $SH_KEEP_TOPK \
        --init_checkpoint "$INIT_CHECKPOINT"

    echo "剪枝完成: $SCENE"
done

# ============================================================================
# Step 2: RD-curve evaluation (multi-checkpoint auto-select)
# ============================================================================
for SCENE in ${SCENES[@]}
do
    YAML_PATH="cfgs/mesongs/$CONFIG/$SCENE.yaml"
    SOURCE_PATH="$DATA_ROOT/$DS/$SCENE"
    OUTPUT_DIR="outputs_autotune/mesongs_plus_${SCENE}_${CONFIG}_euler_rd_curve_nb${N_BLOCK}_bits${NUM_BITS}_cb${CODEBOOK_SIZE}_topk${SH_KEEP_TOPK}_three_newcodec"

    echo "=========================================="
    echo "[Step 2/2] RD eval $SCENE (multi-ckpt auto-select)"
    echo "  Output: $OUTPUT_DIR"
    echo "=========================================="

    PYTHONPATH=$PYTHONPATH:$(pwd) CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python splatwizard/scripts/eval_rd_curve.py \
        --source_path "$SOURCE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --yaml_path "$YAML_PATH" \
        --model mesongs_plus \
        --optim mesongs_plus \
        --eval_mode ENCODE_DECODE \
        --force_setup \
        --scene_imp "$SCENE" \
        --use_quat False \
        --n_block $N_BLOCK \
        --num_bits $NUM_BITS \
        --codebook_size $CODEBOOK_SIZE \
        --raht $RAHT \
        --use_indexed $USE_INDEXED \
        --sh_keep_threshold $SH_KEEP_THRESHOLD \
        --sh_keep_topk $SH_KEEP_TOPK \
        --save_bitstream \
        --save_rendered_image

    echo "RD 评估完成: $SCENE"
done

echo "=========================================="
echo "Mip-NeRF 360 全流程完成！"
echo "=========================================="
