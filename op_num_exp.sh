
#srun -p VI_Face_1080TI -N 1 --gres=gpu:8 --job-name=op3 python -u random_search.py \
#     --opnum 3 --opdepth 2 --policybatch 3 --attackbatch 16 --datasplit 500 --log op_2.log &

#srun -p VI_Face_1080TI -N 1  --gres=gpu:8 --job-name=op5 python -u random_search.py \
#     --opnum 3 --opdepth 3 --policybatch 3 --attackbatch 16 --datasplit 500 --log op_3.log &

srun -p VI_Face_1080TI -N 1 --gres=gpu:8 --job-name=op7 python -u random_search.py \
     --opnum 3 --opdepth 4 --policybatch 3 --attackbatch 16 --datasplit 500 --log op_4.log &

#srun -p VI_Face_1080TI -N 1 --gres=gpu:8 --job-name=op9 python -u random_search.py \
#     --opnum 3 --opdepth 5 --policybatch 3 --attackbatch 16 --datasplit 500 --log op_5.log &