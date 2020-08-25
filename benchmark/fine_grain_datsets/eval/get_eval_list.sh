
python get_eval_list.py \
    --listf "/home/haojieyuan/Data/oxfordFlowers/test_list.txt" \
    --outf "/home/haojieyuan/Data/oxfordFlowers/eval_gt_labels.txt"

cp /home/haojieyuan/Data/oxfordFlowers/eval_gt_labels.txt /home/haojieyuan/Data/oxfordFlowers/eval_target_labels.txt

python get_eval_list.py \
    --listf "/home/haojieyuan/Data/caltech101/test_list.txt" \
    --outf "/home/haojieyuan/Data/caltech101/eval_gt_labels.txt"

cp /home/haojieyuan/Data/caltech101/eval_gt_labels.txt /home/haojieyuan/Data/caltech101/eval_target_labels.txt
