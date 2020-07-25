# cuda_id 8
# BATCH_SIZE=10
# NUM_ITER=10
# MOMENTUM=1.0
# DI_PROB=0
# USE_TI=False
# AUTO_AUGFILE=./autoaug.txt
# OUT_DIR_PREFIX=ours




sh run_attack.sh 8 2 1  0   0.7 False ./autoaug.txt ours_DI_FGSM 1> ./out/ours_di_fgsm_log.log 2>&1
sh run_attack.sh 8 2 10 0   0.7 False ./autoaug.txt ours_DI_IFGSM 1> ./out/ours_di_ifgsm_log.log 2>&1
sh run_attack.sh 8 2 10 1.0 0.7 False ./autoaug.txt ours_DI_MIFGSM 1> ./out/ours_di_mifgsm_log.log 2>&1