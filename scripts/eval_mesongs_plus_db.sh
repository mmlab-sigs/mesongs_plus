#!/bin/bash
# ============================================================================
# MesonGS Plus 全流程（Train + RD 曲线评估） - Deep Blending (db)
# ============================================================================

set -e

eval "$(conda shell.bash hook)"
conda activate compressgs

CONFIG=c1
DS=db

N_BLOCK=80
CODEBOOK_SIZE=4096
NUM_BITS=16
RAHT=True
USE_INDEXED=True
SH_KEEP_THRESHOLD=-1
SH_KEEP_TOPK=1000000

SCENES=('drjohnson' 'playroom')
SCENES=('drjohnson')

DATA_ROOT=${DATA_ROOT:-/mnt/storage/users/szxie_data/nerf_data}
INIT_CKPT_ROOT=${INIT_CKPT_ROOT:-/mnt/storage/users/szxie_data/MesonGS/outputs}
export TMC3_PATH=${TMC3_PATH:-/home/gejunchen/Work/2026-1/Projects/compressgs_autotune/mpeg-pcc-tmc13/build/tmc3/tmc3}

CUDA_DEVICE=${CUDA_DEVICE:-0}

# ----------------------------------------------------------------------------
# Step 1: Pruning (multi-prune)
# ----------------------------------------------------------------------------
for SCENE in ${SCENES[@]}
do
    YAML_PATH="cfgs/mesongs/$CONFIG/$SCENE.yaml"
    SOURCE_PATH="$DATA_ROOT/$DS/$SCENE"
    INIT_CHECKPOINT="$INIT_CKPT_ROOT/$DS/$SCENE/baseline/3dgs/point_cloud/iteration_30000/point_cloud.ply"

    echo "=========================================="
    echo "[Step 1/2] Pruning $SCENE (multi pruning_rates from YAML)"
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
done

# ----------------------------------------------------------------------------
# Step 2: RD-curve evaluation
# ----------------------------------------------------------------------------
for SCENE in ${SCENES[@]}
do
    YAML_PATH="cfgs/mesongs/$CONFIG/$SCENE.yaml"
    SOURCE_PATH="$DATA_ROOT/$DS/$SCENE"
    OUTPUT_DIR="outputs_autotune/mesongs_plus_${SCENE}_${CONFIG}_euler_rd_curve_nb${N_BLOCK}_bits${NUM_BITS}_cb${CODEBOOK_SIZE}_topk${SH_KEEP_TOPK}_three_newcodec"

    echo "=========================================="
    echo "[Step 2/2] RD eval $SCENE"
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
        --golden_search_interval 50000 \
        --pruning_rate -1 \
        --save_bitstream \
        --save_rendered_image
done

echo "=========================================="
echo "Deep Blending 全流程完成！"
echo "=========================================="
