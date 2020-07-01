#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=/home/haojieyuan/Data/ImageNet/nips2017_dev
MAX_EPSILON=16

#OUTPUT_DIR=./TI_DIM_attack_out_ens/
#CUDA_VISIBLE_DEVICES=8 python attack_iter.py \
#  --input_dir="${INPUT_DIR}" \
#  --output_dir="${OUTPUT_DIR}" \
#  --max_epsilon="${MAX_EPSILON}" \
#  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
#  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
#  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
#  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
#  --target_model=ens \
#  --num_iter=10 \
#  --momentum=1.0 \
#  --prob=0.7

OUTPUT_DIR=./TI_DIM_attack_out_resnet/
CUDA_VISIBLE_DEVICES=8 python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=resnet \
  --num_iter=10 \
  --momentum=1.0 \
  --prob=0.7


OUTPUT_DIR=./TI_DIM_attack_out_inception_v3/
CUDA_VISIBLE_DEVICES=8 python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=inception_v3 \
  --num_iter=10 \
  --momentum=1.0 \
  --prob=0.7


OUTPUT_DIR=./TI_DIM_attack_out_inception_v4/
CUDA_VISIBLE_DEVICES=8 python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=inception_v4 \
  --num_iter=10 \
  --momentum=1.0 \
  --prob=0.7


OUTPUT_DIR=./TI_DIM_attack_out_inception_resnet_v2/
CUDA_VISIBLE_DEVICES=8 python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=inception_resnet_v2 \
  --num_iter=10 \
  --momentum=1.0 \
  --prob=0.7