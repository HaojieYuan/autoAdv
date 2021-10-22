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
NUM_ITER=10
MOMENTUM=1.0
DI_PROB=0.0

CUDA_ID=$1
BATCH_SIZE=$2
USE_TI=$3
TI_KERNEL=$4
USE_SI=$5
USE_DEM=$6
USE_NI=$7
AUTO_AUGFILE=$8

OUT_DIR_PREFIX=$9


OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_resnet/
echo $OUTPUT_DIR
time CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=resnet \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --kernel=$TI_KERNEL \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --use_dem=$USE_DEM \
  --use_ni=$USE_NI \
  --autoaug_file="$AUTO_AUGFILE"


OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v3/
echo $OUTPUT_DIR
time CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=inception_v3 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --kernel=$TI_KERNEL \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --use_dem=$USE_DEM \
  --use_ni=$USE_NI \
  --autoaug_file="$AUTO_AUGFILE"


OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v4/
echo $OUTPUT_DIR
time CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=inception_v4 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --kernel=$TI_KERNEL \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --use_dem=$USE_DEM \
  --use_ni=$USE_NI \
  --autoaug_file="$AUTO_AUGFILE"


OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_resnet_v2/
echo $OUTPUT_DIR
time CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=inception_resnet_v2 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --kernel=$TI_KERNEL \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --use_dem=$USE_DEM \
  --use_ni=$USE_NI \
  --autoaug_file="$AUTO_AUGFILE"






#echo "Now evaluating."

#cd /home/haojieyuan/autoAdv/benchmark


#sh run_eval_on_img_dir.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v3
#sh run_eval_on_img_dir.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v4
#sh run_eval_on_img_dir.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_resnet_v2
#sh run_eval_on_img_dir.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_resnet

#echo "Things Done."


