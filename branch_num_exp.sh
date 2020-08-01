srun -p VI_Face_1080TI -N 1 --gres=gpu:8 --job-name=branch1 python -u random_search.py \
     --opnum 1 --opdepth 1 --policybatch 5 --attackbatch 32 --datasplit 500 --log branch_1.log &

srun -p VI_Face_1080TI -N 1 --gres=gpu:8 --job-name=branch3 python -u random_search.py \
     --opnum 3 --opdepth 1 --policybatch 5 --attackbatch 16 --datasplit 500 --log branch_3.log &

srun -p VI_Face_1080TI -N 1  --gres=gpu:8 --job-name=branch5 python -u random_search.py \
     --opnum 5 --opdepth 1 --policybatch 5 --attackbatch 8 --datasplit 500 --log branch_5.log &

srun -p VI_Face_1080TI -N 1 --gres=gpu:8 --job-name=branch7 python -u random_search.py \
     --opnum 7 --opdepth 1 --policybatch 5 --attackbatch 8 --datasplit 500 --log branch_7.log &

srun -p VI_Face_1080TI -N 1 --gres=gpu:8 --job-name=branch9 python -u random_search.py \
     --opnum 9 --opdepth 1 --policybatch 5 --attackbatch 4 --datasplit 500 --log branch_9.log &