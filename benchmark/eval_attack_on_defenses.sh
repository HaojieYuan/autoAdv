
CUDA_ID=$1
ATTACK=$2

source /home/haojieyuan/.bashrc
pytorch
bash ./1stHGD/Guided-Denoise-master/nips_deploy/run_defense.sh $CUDA_ID $ATTACK

source /home/haojieyuan/.bashrc
ca tf
cuda9
bash ./2nd/NIPS2017_adv_challenge_defense/run_defense.sh $CUDA_ID $ATTACK
bash ./3rd/mmd/run_defense.sh $CUDA_ID $ATTACK
PATH=$(getconf PATH)
unset LD_LIBRARY_PATH