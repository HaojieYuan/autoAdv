
echo "Evaluating Inception v3 on Oxford Flowers."
CUDA_VISIBLE_DEVICES=1 python eval.py --model_name="inception_v3" \
  --checkpoint_path="/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/ckpts/oxfordFlowers_adam/" \
  --batch_size=50 \
  --dataset_name="oxfordFlowers" \
  --num_classes=102 \
  --split 'train'

echo "Evaluating Inception v3 on Oxford Flowers."
CUDA_VISIBLE_DEVICES=1 python eval.py --model_name="inception_v3" \
  --checkpoint_path="/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/ckpts/oxfordFlowers_adam/" \
  --batch_size=50 \
  --dataset_name="oxfordFlowers" \
  --num_classes=102 \
  --split 'test'

exit 0

echo "Evaluating Inception v3 on FGVC Aircraft."
CUDA_VISIBLE_DEVICES=1 python eval.py --model_name="inception_v3" \
  --checkpoint_path="/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/ckpts/fgvcAircraft/" \
  --batch_size=50 \
  --dataset_name="fgvcAircraft" \
  --num_classes=100

echo "Evaluating Inception v3 on Stanford Cars."
CUDA_VISIBLE_DEVICES=1 python eval.py --model_name="inception_v3" \
  --checkpoint_path="/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/ckpts/stanfordCars/" \
  --batch_size=50 \
  --dataset_name="stanfordCars" \
  --num_classes=196