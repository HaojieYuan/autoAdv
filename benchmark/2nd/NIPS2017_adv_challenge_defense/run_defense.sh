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


echo "2nd eval."

CUDA_VISIBLE_DEVICES=$CUDA_ID python defense.py \
  --input_dir="../../attacks/TI/outv1/${INPUT_DIR}_inception_v3/" \
  --output_file="${INPUT_DIR}_inception_v3.txt" \
  --checkpoint_path=ens_adv_inception_resnet_v2.ckpt


CUDA_VISIBLE_DEVICES=$CUDA_ID python defense.py \
  --input_dir="../../attacks/TI/outv1/${INPUT_DIR}_inception_v4/" \
  --output_file="${INPUT_DIR}_inception_v4.txt" \
  --checkpoint_path=ens_adv_inception_resnet_v2.ckpt


CUDA_VISIBLE_DEVICES=$CUDA_ID python defense.py \
  --input_dir="../../attacks/TI/outv1/${INPUT_DIR}_inception_resnet_v2/" \
  --output_file="${INPUT_DIR}_inception_resnet_v2.txt" \
  --checkpoint_path=ens_adv_inception_resnet_v2.ckpt


CUDA_VISIBLE_DEVICES=$CUDA_ID python defense.py \
  --input_dir="../../attacks/TI/outv1/${INPUT_DIR}_resnet/" \
  --output_file="${INPUT_DIR}_resnet.txt" \
  --checkpoint_path=ens_adv_inception_resnet_v2.ckpt

  echo "2nd eval done."