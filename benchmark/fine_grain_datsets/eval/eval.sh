CUDA_ID=$1
IMG_DIR=$2
LPREFIX=$3
CKPT_PATH=$4

TMP_NAME=$(basename ${IMG_DIR})
TMP_OUT=.${TMP_NAME}.tfrecords
CUDA_VISIBLE_DEVICES=CUDA_ID python transform2TFrecords.py --folder $IMG_DIR --out $TMP_OUT\
    --lprefix $LPREFIX

MODEL_NAME=inception_v4
CHECKPOINT_PATH=$CKPT_PATH
#/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v4/model.ckpt-6375
CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT

rm $TMP_OUT