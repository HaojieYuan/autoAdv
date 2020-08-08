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
#sh run_attack.sh 8 2 1  0   0.7 False ./autoaug.txt ours_DI_FGSM 1> ./out/ours_di_fgsm_log.log 2>&1
#sh run_attack.sh 8 2 10 0   0.7 False ./autoaug.txt ours_DI_IFGSM 1> ./out/ours_di_ifgsm_log.log 2>&1
#sh run_attack.sh 8 2 10 1.0 0.7 False ./autoaug.txt ours_DI_MIFGSM 1> ./out/ours_di_mifgsm_log.log 2>&1