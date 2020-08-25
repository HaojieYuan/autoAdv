CUDA_ID=5
MAX_EPSILON=16

NUM_ITER=10
MOMENTUM=1.0




# attack on Caltech101
INPUT_DIR=/home/haojieyuan/Data/caltech101/eval_imgs_resized


# using DI+TI-MI
OUT_DIR_PREFIX=FGVC_Caltech_ours_avg_TI_MI_FGSM
BATCH_SIZE=3
USE_TI=True
USE_SI=False
DI_PROB=0
AUTO_AUGFILE=/home/haojieyuan/autoAdv/benchmark/attacks/TI/autoaug_op3_avg.txt

OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v3/
CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v3/model.ckpt-9563 \
  --target_model=inception_v3 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --autoaug_file="$AUTO_AUGFILE" \
  --FGVC_eval=True \
  --class_num=102

# attack on OxfordFlowers
INPUT_DIR=/home/haojieyuan/Data/oxfordFlowers/eval_imgs_resized

# using DI+TI-MI
OUT_DIR_PREFIX=FGVC_Flowers_ours_avg_TI_MI_FGSM
BATCH_SIZE=3
USE_TI=True
USE_SI=False
DI_PROB=0
AUTO_AUGFILE=/home/haojieyuan/autoAdv/benchmark/attacks/TI/autoaug_op3_avg.txt

OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v3/
CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v3/model.ckpt-6375\
  --target_model=inception_v3 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --autoaug_file="$AUTO_AUGFILE" \
  --FGVC_eval=True \
  --class_num=102





cd /home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval
echo "===========Evaluating DI+TI-MI-FGSM.========="
OUT_DIR_PREFIX=FGVC_Caltech_ours_avg_TI_MI_FGSM
sh eval.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v3 \
   "/home/haojieyuan/Data/caltech101/eval_" \
   "/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v4/model.ckpt-9563"


echo "===========Evaluating DI+TI-MI-FGSM.========="
OUT_DIR_PREFIX=FGVC_Flowers_ours_avg_TI_MI_FGSM
sh eval.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v3 \
   "/home/haojieyuan/Data/oxfordFlowers/eval_" \
   "/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v4/model.ckpt-6375"




exit 0



################################################################################
CUDA_ID=4
MAX_EPSILON=16

NUM_ITER=10
MOMENTUM=1.0




# attack on Caltech101
INPUT_DIR=/home/haojieyuan/Data/caltech101/eval_imgs_resized

# using TI-MI
OUT_DIR_PREFIX=FGVC_Caltech_TI_MI_FGSM
BATCH_SIZE=10
USE_TI=True
USE_SI=False
DI_PROB=0
AUTO_AUGFILE=None

OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v3/
CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v3/model.ckpt-9563 \
  --target_model=inception_v3 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --autoaug_file="$AUTO_AUGFILE" \
  --FGVC_eval=True \
  --class_num=102


# using DI+TI-MI
OUT_DIR_PREFIX=FGVC_Caltech_DI_TI_MI_FGSM
BATCH_SIZE=10
USE_TI=True
USE_SI=False
DI_PROB=0.7
AUTO_AUGFILE=None

OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v3/
CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v3/model.ckpt-9563 \
  --target_model=inception_v3 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --autoaug_file="$AUTO_AUGFILE" \
  --FGVC_eval=True \
  --class_num=102


# using SI+TI-MI
OUT_DIR_PREFIX=FGVC_Caltech_SI_TI_MI_FGSM
BATCH_SIZE=10
USE_TI=True
USE_SI=True
DI_PROB=0
AUTO_AUGFILE=None

OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v3/
CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v3/model.ckpt-9563 \
  --target_model=inception_v3 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --autoaug_file="$AUTO_AUGFILE" \
  --FGVC_eval=True \
  --class_num=102


# using ours+TI-MI
OUT_DIR_PREFIX=FGVC_Caltech_ours_TI_MI_FGSM
BATCH_SIZE=3
USE_TI=True
USE_SI=False
DI_PROB=0
AUTO_AUGFILE='./autoaug_op3.txt'

OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v3/
CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v3/model.ckpt-9563 \
  --target_model=inception_v3 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --autoaug_file="$AUTO_AUGFILE" \
  --FGVC_eval=True \
  --class_num=102









# attack on OxfordFlowers
INPUT_DIR=/home/haojieyuan/Data/oxfordFlowers/eval_imgs_resized

# using TI-MI
OUT_DIR_PREFIX=FGVC_Flowers_TI_MI_FGSM
BATCH_SIZE=10
USE_TI=True
USE_SI=False
DI_PROB=0
AUTO_AUGFILE=None

OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v3/
CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v3/model.ckpt-6375 \
  --target_model=inception_v3 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --autoaug_file="$AUTO_AUGFILE" \
  --FGVC_eval=True \
  --class_num=102


# using DI+TI-MI
OUT_DIR_PREFIX=FGVC_Flowers_DI_TI_MI_FGSM
BATCH_SIZE=10
USE_TI=True
USE_SI=False
DI_PROB=0.7
AUTO_AUGFILE=None

OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v3/
CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v3/model.ckpt-6375\
  --target_model=inception_v3 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --autoaug_file="$AUTO_AUGFILE" \
  --FGVC_eval=True \
  --class_num=102


# using SI+TI-MI
OUT_DIR_PREFIX=FGVC_Flowers_SI_TI_MI_FGSM
BATCH_SIZE=10
USE_TI=True
USE_SI=True
DI_PROB=0
AUTO_AUGFILE=None

OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v3/
CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v3/model.ckpt-6375 \
  --target_model=inception_v3 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --autoaug_file="$AUTO_AUGFILE" \
  --FGVC_eval=True \
  --class_num=102


# using ours+TI-MI
OUT_DIR_PREFIX=FGVC_Flowers_ours_TI_MI_FGSM
BATCH_SIZE=3
USE_TI=True
USE_SI=False
DI_PROB=0
AUTO_AUGFILE='./autoaug_op3.txt'

OUTPUT_DIR=./out/${OUT_DIR_PREFIX}_inception_v3/
CUDA_VISIBLE_DEVICES=$CUDA_ID python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --checkpoint_path_inception_v3=/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v3/model.ckpt-6375 \
  --target_model=inception_v3 \
  --batch_size=$BATCH_SIZE \
  --num_iter=$NUM_ITER \
  --momentum=$MOMENTUM \
  --use_ti=$USE_TI \
  --use_si=$USE_SI \
  --prob=$DI_PROB \
  --autoaug_file="$AUTO_AUGFILE" \
  --FGVC_eval=True \
  --class_num=102






cd /home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval
echo "===========Evaluating TI-MI-FGSM.========="
OUT_DIR_PREFIX=FGVC_Caltech_TI_MI_FGSM
sh eval.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v3 \
   "/home/haojieyuan/Data/caltech101/eval_" \
   "/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v4/model.ckpt-9563"

echo "===========Evaluating DI+TI-MI-FGSM.========="
OUT_DIR_PREFIX=FGVC_Caltech_DI_TI_MI_FGSM
sh eval.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v3 \
   "/home/haojieyuan/Data/caltech101/eval_" \
   "/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v4/model.ckpt-9563"

echo "===========Evaluating SI+TI-MI-FGSM.========="
OUT_DIR_PREFIX=FGVC_Caltech_SI_TI_MI_FGSM
sh eval.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v3 \
   "/home/haojieyuan/Data/caltech101/eval_" \
   "/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v4/model.ckpt-9563"

echo "===========Evaluating ours+TI-MI-FGSM.========="
OUT_DIR_PREFIX=FGVC_Caltech_ours_TI_MI_FGSM
sh eval.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v3 \
   "/home/haojieyuan/Data/caltech101/eval_" \
   "/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/caltech101/inception_v4/model.ckpt-9563"





echo "===========Evaluating TI-MI-FGSM.========="
OUT_DIR_PREFIX=FGVC_Flowers_TI_MI_FGSM
sh eval.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v3 \
   "/home/haojieyuan/Data/oxfordFlowers/eval_" \
   "/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v4/model.ckpt-6375"

echo "===========Evaluating DI+TI-MI-FGSM.========="
OUT_DIR_PREFIX=FGVC_Flowers_DI_TI_MI_FGSM
sh eval.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v3 \
   "/home/haojieyuan/Data/oxfordFlowers/eval_" \
   "/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v4/model.ckpt-6375"

echo "===========Evaluating SI+TI-MI-FGSM.========="
OUT_DIR_PREFIX=FGVC_Flowers_SI_TI_MI_FGSM
sh eval.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v3 \
   "/home/haojieyuan/Data/oxfordFlowers/eval_" \
   "/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v4/model.ckpt-6375"

echo "===========Evaluating ours+TI-MI-FGSM.========="
OUT_DIR_PREFIX=FGVC_Flowers_ours_TI_MI_FGSM
sh eval.sh $CUDA_ID /home/haojieyuan/autoAdv/benchmark/attacks/TI/out/${OUT_DIR_PREFIX}_inception_v3 \
   "/home/haojieyuan/Data/oxfordFlowers/eval_" \
   "/home/haojieyuan/autoAdv/benchmark/fine_grain_datsets/eval/models/oxfordFlowers/inception_v4/model.ckpt-6375"
