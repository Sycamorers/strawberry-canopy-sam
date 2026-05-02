#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_SCRIPT_PATH="scripts/run_canopy_benchmark.py"

POINTCASES=("G")
METHODS=("2p_8")
PRE_MASKS=("with_pre_mask")
BOX_TYPES=("overlapbox")
LOCATIONS=("GNV")
MOBV2_ENCODER_TYPE=("efficientvit_l2")
WHICH_SAM=("SAM")
WHICH_EFFICIENT_SAM=("EfficientSAM_s")
NEWMETHODS=("true")

declare -A BASE_DIRS=( ["GNV"]="outputs/GNV_benchmark_results" )
declare -A IMAGE_FOLDERS=( ["GNV"]="examples/GNV_benchmark_data_coco" )
declare -A ANNOTATIONS_PATHS=( ["GNV"]="examples/GNV_benchmark_data_coco/annotations.json" )
declare -A IOU_OUTPUT_DIRS=( ["GNV"]="outputs/iou_results/GNV" )

for LOCATION in "${LOCATIONS[@]}"; do
    BASE_DIR=${BASE_DIRS[$LOCATION]}
    IMAGE_FOLDER=${IMAGE_FOLDERS[$LOCATION]}
    ANNOTATIONS_PATH=${ANNOTATIONS_PATHS[$LOCATION]}
    IOU_OUTPUT_DIR=${IOU_OUTPUT_DIRS[$LOCATION]}

    for POINTCASE in "${POINTCASES[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            for BOX_TYPE in "${BOX_TYPES[@]}"; do
                for NEWMETHOD in "${NEWMETHODS[@]}"; do
                    if [ "$NEWMETHOD" = "true" ]; then
                        CURRENT_PRE_MASKS=("with_pre_mask")
                    else
                        CURRENT_PRE_MASKS=("${PRE_MASKS[@]}")
                    fi

                    for PRE_MASK in "${CURRENT_PRE_MASKS[@]}"; do
                        if [ "$PRE_MASK" = "no_pre_mask" ] && [ "$NEWMETHOD" = "true" ]; then
                            continue
                        fi

                        if [ "$PRE_MASK" = "with_pre_mask" ]; then
                            USE_PRELIMINARY_MASKS="true"
                        else
                            USE_PRELIMINARY_MASKS="false"
                        fi

                        if [ "$METHOD" = "0p" ]; then
                            USE_EXCLUSIVE_POINTS="false"
                        else
                            USE_EXCLUSIVE_POINTS="true"
                        fi

                        python "$PYTHON_SCRIPT_PATH" \
                            --pointcase "$POINTCASE" \
                            --method "$METHOD" \
                            --pre_mask "$PRE_MASK" \
                            --box_type "$BOX_TYPE" \
                            --use_preliminary_masks "$USE_PRELIMINARY_MASKS" \
                            --use_exclusive_points "$USE_EXCLUSIVE_POINTS" \
                            --base_dir "$BASE_DIR" \
                            --image_folder "$IMAGE_FOLDER" \
                            --which_sam "$WHICH_SAM" \
                            --which_efficient_sam "$WHICH_EFFICIENT_SAM" \
                            --mobv2_encoder_type "$MOBV2_ENCODER_TYPE" \
                            --annotations_path "$ANNOTATIONS_PATH" \
                            --iou_output_dir "$IOU_OUTPUT_DIR" \
                            --method_name "$METHOD" \
                            --newmethod "$NEWMETHOD"
                    done
                done
            done
        done
    done
done
