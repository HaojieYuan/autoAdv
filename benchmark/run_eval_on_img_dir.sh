
#IMG_DIR=/home/haojieyuan/autoAdv/benchmark/attacks/DI/M-DI-2-FGSM_attack_out_inception_resnet_v2
#IMG_DIR=/home/haojieyuan/autoAdv/benchmark/attacks/TI/ours_TI_DIM_ens
CUDA_ID=$1
IMG_DIR=$2

TMP_NAME=$(basename ${IMG_DIR})
TMP_OUT=.${TMP_NAME}.tfrecords
CUDA_VISIBLE_DEVICES=$CUDA_ID python transform2TFrecords.py --folder $IMG_DIR --out $TMP_OUT




MODEL_NAME=inception_v3
CHECKPOINT_PATH=./pretrained/ensemble/ens3_adv_inception_v3.ckpt

CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT



MODEL_NAME=inception_v3
CHECKPOINT_PATH=./pretrained/ensemble/ens4_adv_inception_v3.ckpt

CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT




MODEL_NAME=inception_resnet_v2
CHECKPOINT_PATH=./pretrained/ensemble/ens_adv_inception_resnet_v2.ckpt

CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT


MODEL_NAME=inception_v3
CHECKPOINT_PATH=./pretrained/normal/inception_v3.ckpt

CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT


MODEL_NAME=inception_v4
CHECKPOINT_PATH=./pretrained/normal/inception_v4.ckpt

CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT


MODEL_NAME=inception_resnet_v2
CHECKPOINT_PATH=./pretrained/normal/inception_resnet_v2_2016_08_30.ckpt

CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT


MODEL_NAME=resnet_v2
CHECKPOINT_PATH=./pretrained/normal/resnet_v2_152.ckpt

CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT

rm $TMP_OUT