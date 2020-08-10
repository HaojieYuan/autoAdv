# cuda_id 8
# BATCH_SIZE=10
# NUM_ITER=10
# MOMENTUM=1.0
# DI_PROB=0.7
# USE_TI=False
# USE_SI=False
# AUTO_AUGFILE=./autoaug.txt or None
# OUT_DIR_PREFIX=ours



#sh run_attack.sh 8 10 10 1.0 0   True False None TI_MI_FGSM    1> ./out/ti_mi_fgsm_log.log 2>&1
#echo "TI-MI Done."
#sh run_attack.sh 8 2  10 1.0 0   True True  None SI_TI_MI_FGSM 1> ./out/si_ti_mi_fgsm_log.log 2>&1
#echo "SI-TI-MI Done."

#sh run_attack.sh 8 10 10 1.0 0 False False ./autoaug_b1.txt ours_MI_FGSM_b1_ti7
#exit 0

sh run_attack.sh 8 10 10 1.0 0 False False ./autoaug_b1.txt ours_MI_FGSM_b1_ti7 1> ./out/ours_MI_FGSM_b1_ti7_log.log 2>&1
sh run_attack.sh 8 10 10 1.0 0 True  Fasle ./autoaug_b1.txt ours_TI_MI_FGSM_b1_ti7 1> ./out/ours_TI_MI_FGSM_b1_ti7_log.log 2>&1

sh run_attack.sh 8 3 10 1.0 0 False False ./autoaug_b3.txt ours_MI_FGSM_b3_ti7 1> ./out/ours_MI_FGSM_b3_ti7_log.log 2>&1
sh run_attack.sh 8 3 10 1.0 0 True  False ./autoaug_b3.txt ours_TI_MI_FGSM_b3_ti7 1> ./out/ours_TI_MI_FGSM_b3_ti7_log.log 2>&1

sh run_attack.sh 8 1 10 1.0 0 False False ./autoaug_b5.txt ours_MI_FGSM_b5_ti7 1> ./out/ours_MI_FGSM_b5_ti7_log.log 2>&1
sh run_attack.sh 8 1 10 1.0 0 True  False ./autoaug_b5.txt ours_TI_MI_FGSM_b5_ti7 1> ./out/ours_TI_MI_FGSM_b5_ti7_log.log 2>&1

sh run_attack.sh 8 1 10 1.0 0 False False ./autoaug_b7.txt ours_MI_FGSM_b7_ti7 1> ./out/ours_MI_FGSM_b7_ti7_log.log 2>&1
sh run_attack.sh 8 1 10 1.0 0 True  False ./autoaug_b7.txt ours_TI_MI_FGSM_b7_ti7 1> ./out/ours_TI_MI_FGSM_b7_ti7_log.log 2>&1

sh run_attack.sh 8 1 10 1.0 0 False False ./autoaug_b9.txt ours_MI_FGSM_b9_ti7 1> ./out/ours_MI_FGSM_b9_ti7_log.log 2>&1
sh run_attack.sh 8 1 10 1.0 0 True  False ./autoaug_b9.txt ours_TI_MI_FGSM_b9_ti7 1> ./out/ours_TI_MI_FGSM_b9_ti7_log.log 2>&1

