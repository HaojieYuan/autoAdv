#!/bin/bash
#
# run_defense.sh is a script which executes the defense
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_defense.sh INPUT_DIR OUTPUT_FILE
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_FILE - file to store classification labels
#

CUDA_ID=$1
INPUT_DIR=$2
work_path=$(dirname $0)

echo "3rd eval."


CUDA_VISIBLE_DEVICES=$CUDA_ID python $work_path/defense_mmd.py \
  --input_dir="${work_path}/../../attacks/TI/out/${INPUT_DIR}_inception_v3/" \
  --output_file="${work_path}/${INPUT_DIR}_inception_v3.txt"


CUDA_VISIBLE_DEVICES=$CUDA_ID python $work_path/defense_mmd.py \
  --input_dir="${work_path}/../../attacks/TI/out/${INPUT_DIR}_inception_v4/" \
  --output_file="${work_path}/${INPUT_DIR}_inception_v4.txt"


CUDA_VISIBLE_DEVICES=$CUDA_ID python $work_path/defense_mmd.py \
  --input_dir="${work_path}/../../attacks/TI/out/${INPUT_DIR}_inception_resnet_v2/" \
  --output_file="${work_path}/${INPUT_DIR}_inception_resnet_v2.txt"


CUDA_VISIBLE_DEVICES=$CUDA_ID python $work_path/defense_mmd.py \
  --input_dir="${work_path}/../../attacks/TI/out/${INPUT_DIR}_resnet/" \
  --output_file="${work_path}/${INPUT_DIR}_resnet.txt"

echo '3rd eval done.'