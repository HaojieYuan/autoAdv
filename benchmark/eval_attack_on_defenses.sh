
CUDA_ID=$1
ATTACK=$2

sh ./1stHGD/Guided-Denoise-master/nips_deploy/run_defense.sh $CUDA_ID $ATTACK
sh ./2nd/NIPS2017_adv_challenge_defense/run_defense.sh $CUDA_ID $ATTACK
sh ./3rd/mmd/run_defense.sh $CUDA_ID $ATTACK