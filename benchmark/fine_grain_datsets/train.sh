CUDA_VISIBLE_DEVICES=8 python train_model.py \
    --batch_size 64 \
    --datasetname "oxfordFlowers" \
    --model_name "inception_v3" \
    --train_dir "./ckpts/oxfordFlowers" \
    --max_number_of_steps 10000  1> ./log/oxfordFlowers.log 2>&1

CUDA_VISIBLE_DEVICES=8 python train_model.py \
    --batch_size 64 \
    --datasetname "fgvcAircraft" \
    --model_name "inception_v3" \
    --train_dir "./ckpts/fgvcAircraft" \
    --max_number_of_steps 10000  1> ./log/fgvcAircraft.log 2>&1


CUDA_VISIBLE_DEVICES=8 python train_model.py \
    --batch_size 64 \
    --datasetname "stanfordCars" \
    --model_name "inception_v3" \
    --train_dir "./ckpts/stanfordCars" \
    --max_number_of_steps 10000  1> ./log/stanfordCars.log 2>&1