
#IMG_DIR=/home/haojieyuan/autoAdv/benchmark/attacks/DI/M-DI-2-FGSM_attack_out_inception_resnet_v2
#IMG_DIR=/home/haojieyuan/autoAdv/benchmark/attacks/TI/ours_TI_DIM_ens
CUDA_ID=4
IMG_DIR=/home/haojieyuan/Data/oxfordFlowers/eval_imgs_resized

TMP_NAME=$(basename ${IMG_DIR})
TMP_OUT=.${TMP_NAME}.tfrecords
CUDA_VISIBLE_DEVICES=CUDA_ID python transform2TFrecords.py --folder $IMG_DIR --out $TMP_OUT\
    --lprefix /home/haojieyuan/Data/oxfordFlowers/eval_


MODEL_NAME=inception_v3
CHECKPOINT_PATH=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v3/model.ckpt-6375

CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT



MODEL_NAME=inception_v4
CHECKPOINT_PATH=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v4/model.ckpt-6375

CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT

rm $TMP_OUT



CUDA_ID=4
IMG_DIR=/home/haojieyuan/Data/caltech101/eval_imgs_resized

TMP_NAME=$(basename ${IMG_DIR})
TMP_OUT=.${TMP_NAME}.tfrecords
CUDA_VISIBLE_DEVICES=CUDA_ID python transform2TFrecords.py --folder $IMG_DIR --out $TMP_OUT\
    --lprefix /home/haojieyuan/Data/caltech101/eval_



MODEL_NAME=inception_v3
CHECKPOINT_PATH=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v3/model.ckpt-9563

CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT



MODEL_NAME=inception_v4
CHECKPOINT_PATH=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v4/model.ckpt-9563

CUDA_VISIBLE_DEVICES=$CUDA_ID python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_tfrecords=$TMP_OUT

rm $TMP_OUT



