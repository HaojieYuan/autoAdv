# 1000 * 2040/32 = 63750
CUDA_VISIBLE_DEVICES=8 python train_model.py \
    --batch_size 32 \
    --dataset_name "oxfordFlowers" \
    --model_name "inception_v3" \
    --train_dir "./ckpts/oxfordFlowers" \
    --max_number_of_steps 63750  1> ./log/oxfordFlowers_RMSProp.log 2>&1

# 1000 * 6667/32 = 208343.75
CUDA_VISIBLE_DEVICES=8 python train_model.py \
    --batch_size 32 \
    --dataset_name "stanfordCars" \
    --model_name "inception_v3" \
    --train_dir "./ckpts/stanfordCars" \
    --max_number_of_steps 208344  1> ./log/stanfordCars_RMSProp.log 2>&1

# 1000 * 8144/32 = 254500
CUDA_VISIBLE_DEVICES=8 python train_model.py \
    --batch_size 32 \
    --dataset_name "fgvcAircraft" \
    --model_name "inception_v3" \
    --train_dir "./ckpts/fgvcAircraft" \
    --max_number_of_steps 254500  1> ./log/fgvcAircraft_RMSProp.log 2>&1


