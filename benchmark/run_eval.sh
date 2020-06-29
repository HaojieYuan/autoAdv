MODEL_NAME=inception_v3
#CHECKPOINT_PATH=./pretrained/ensemble/ens3_adv_inception_v3.ckpt
CHECKPOINT_PATH=./pretrained/ensemble/ens4_adv_inception_v3.ckpt

#MODEL_NAME=inception_resnet_v2
#CHECKPOINT_PATH=./pretrained/ensemble/ens_adv_inception_resnet_v2.ckpt

CUDA_VISIBLE_DEVICES=8 python eval.py --model_name=$MODEL_NAME \
  --checkpoint_path=$CHECKPOINT_PATH \
  --batch_size=50 \
  --test_part=test
