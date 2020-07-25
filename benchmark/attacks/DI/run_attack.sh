
INPUT_DIR=/home/haojieyuan/Data/ImageNet/nips2017_dev
MAX_EPSILON=16

OUTPUT_DIR=./M-DI-2-FGSM_attack_out_ens/
CUDA_VISIBLE_DEVICES=8 python attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=ens \
  --momentum=1.0

OUTPUT_DIR=./M-DI-2-FGSM_attack_out_resnet/
CUDA_VISIBLE_DEVICES=8 python attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=resnet \
  --momentum=1.0

OUTPUT_DIR=./M-DI-2-FGSM_attack_out_inception_v3/
CUDA_VISIBLE_DEVICES=8 python attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=inception_v3 \
  --momentum=1.0

OUTPUT_DIR=./M-DI-2-FGSM_attack_out_inception_v4/
CUDA_VISIBLE_DEVICES=8 python attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=inception_v4 \
  --momentum=1.0

OUTPUT_DIR=./M-DI-2-FGSM_attack_out_inception_resnet_v2/
CUDA_VISIBLE_DEVICES=8 python attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v3.ckpt \
  --checkpoint_path_inception_v4=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_v4.ckpt\
  --checkpoint_path_inception_resnet_v2=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/inception_resnet_v2_2016_08_30.ckpt \
  --checkpoint_path_resnet=/home/haojieyuan/autoAdv/benchmark/pretrained/normal/resnet_v2_152.ckpt \
  --target_model=inception_resnet_v2 \
  --momentum=1.0