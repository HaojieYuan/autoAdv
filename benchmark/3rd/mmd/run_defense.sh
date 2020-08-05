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

INPUT_DIR=/home/haojieyuan/Data/ImageNet/nips2017_dev
OUTPUT_FILE=./out_list.txt

CUDA_VISIBLE_DEVICES=8 python defense_mmd.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}"
