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

#sh run_attack.sh 8 10 10 1.0 0 False False ./autoaug_b1.txt ours_MI_FGSM_b1_ti7 1> ./out/ours_MI_FGSM_b1_ti7_log.log 2>&1
#sh run_attack.sh 4 10 10 1.0 0 True  False ./autoaug_b1.txt ours_TI_MI_FGSM_b1_ti15 1> ./out/ours_TI_MI_FGSM_b1_ti15_log.log 2>&1

#sh run_attack.sh 8 3 10 1.0 0 False False ./autoaug_b3.txt ours_MI_FGSM_b3_ti7 1> ./out/ours_MI_FGSM_b3_ti7_log.log 2>&1
#sh run_attack.sh 5 3 10 1.0 0 True  False ./autoaug_b3.txt ours_TI_MI_FGSM_b3_ti15 1> ./out/ours_TI_MI_FGSM_b3_ti15_log.log 2>&1

#sh run_attack.sh 8 1 10 1.0 0 False False ./autoaug_b5.txt ours_MI_FGSM_b5_ti7 1> ./out/ours_MI_FGSM_b5_ti7_log.log 2>&1
#sh run_attack.sh 6 1 10 1.0 0 True  False ./autoaug_b5.txt ours_TI_MI_FGSM_b5_ti15 1> ./out/ours_TI_MI_FGSM_b5_ti15_log.log 2>&1

#sh run_attack.sh 8 1 10 1.0 0 False False ./autoaug_b7.txt ours_MI_FGSM_b7_ti7 1> ./out/ours_MI_FGSM_b7_ti7_log.log 2>&1
#sh run_attack.sh 7 1 10 1.0 0 True  False ./autoaug_b7.txt ours_TI_MI_FGSM_b7_ti15 1> ./out/ours_TI_MI_FGSM_b7_ti15_log.log 2>&1

#sh run_attack.sh 8 1 10 1.0 0 False False ./autoaug_b9.txt ours_MI_FGSM_b9_ti7 1> ./out/ours_MI_FGSM_b9_ti7_log.log 2>&1
#sh run_attack.sh 8 1 10 1.0 0 True  False ./autoaug_b9.txt ours_TI_MI_FGSM_b9_ti15 1> ./out/ours_TI_MI_FGSM_b9_ti15_log.log 2>&1

# MI-FGSM
#sh run_attack.sh 4 10 10 1.0 0   False  False None MI_FGSM 1> ./out/MI_FGSM.log 2>&1 &
# DI-MI-FGSM
#sh run_attack.sh 5 10 10 1.0 0.7 False  False None DI_MI_FGSM 1> ./out/DI_MI_FGSM.log 2>&1 &
# SI-MI-FGSM
#sh run_attack.sh 6 2  10 1.0 0   False  True  None SI_MI_FGSM 1> ./out/SI_MI_FGSM.log 2>&1 &

# TI-MI-FGSM
#sh run_attack.sh 4 10 10 1.0 0   True   False None TI15_MI_FGSM 1> ./out/TI15_MI_FGSM.log 2>&1 &
# DI-TI-MI-FGSM
#sh run_attack.sh 5 10 10 1.0 0.7 True   False None DI_TI15_MI_FGSM 1> ./out/DI_TI15_MI_FGSM.log 2>&1 &
# SI-TI-MI-FGSM
#sh run_attack.sh 6 2  10 1.0 0   True   True  None SI_TI15_MI_FGSM 1> ./out/SI_TI15_MI_FGSM.log 2>&1 &


# ours branch num
# MI
#sh run_attack.sh 4 10 10 1.0 0 False False ./autoaug_b1.txt ours_b1_MI_FGSM 1> ./out/ours_b1_MI_FGSM.log 2>&1 &
#sh run_attack.sh 4 3  10 1.0 0 False False ./autoaug_b3_avg.txt ours_b3_avg_MI_FGSM 1> ./out/ours_b3_avg_MI_FGSM.log 2>&1 &
#sh run_attack.sh 5 1  10 1.0 0 False False ./autoaug_b5_avg.txt ours_b5_avg_MI_FGSM 1> ./out/ours_b5_avg_MI_FGSM.log 2>&1 &
#sh run_attack.sh 6 1  10 1.0 0 False False ./autoaug_b7_avg.txt ours_b7_avg_MI_FGSM 1> ./out/ours_b7_avg_MI_FGSM.log 2>&1 &
#sh run_attack.sh 8 1  10 1.0 0 False False ./autoaug_b9.txt ours_b9_MI_FGSM 1> ./out/ours_b9_MI_FGSM.log 2>&1 &
# TI
#sh run_attack.sh 7 10 10 1.0 0 True False ./autoaug_b1.txt ours_b1_TI7_MI_FGSM 1> ./out/ours_b1_TI15_MI_FGSM.log 2>&1 &
#sh run_attack.sh 7 3  10 1.0 0 True False ./autoaug_b3_avg.txt ours_b3_avg_TI15_MI_FGSM 1> ./out/ours_b3_avg_TI15_MI_FGSM.log 2>&1 &
#sh run_attack.sh 8 1  10 1.0 0 True False ./autoaug_b5_avg.txt ours_b5_avg_TI15_MI_FGSM 1> ./out/ours_b5_avg_TI15_MI_FGSM.log 2>&1 &
#sh run_attack.sh 9 1  10 1.0 0 True False ./autoaug_b7_avg.txt ours_b7_avg_TI15_MI_FGSM 1> ./out/ours_b7_avg_TI15_MI_FGSM.log 2>&1 &
#sh run_attack.sh 5 1  10 1.0 0 True False ./autoaug_b9.txt ours_b9_TI7_MI_FGSM 1> ./out/ours_b9_TI15_MI_FGSM.log 2>&1 &

# ours OP num
# MI
#sh run_attack.sh 4 3 10 1.0 0 False  False ./autoaug_op2_avg.txt ours_op2_avg_MI_FGSM 1> ./out/ours_op2_avg_MI_FGSM.log 2>&1 &
#sh run_attack.sh 5 3 10 1.0 0 False  False ./autoaug_op3_avg.txt ours_op3_avg_MI_FGSM 1> ./out/ours_op3_avg_MI_FGSM.log 2>&1 &

#sh run_attack.sh 6 3 10 1.0 0 False  False ./autoaug_op4_avg.txt ours_op4_avg_MI_FGSM 1> ./out/ours_op4_avg_MI_FGSM.log 2>&1 &
#sh run_attack.sh 7 3 10 1.0 0 False  False ./autoaug_op5_avg.txt ours_op5_avg_MI_FGSM 1> ./out/ours_op5_avg_MI_FGSM.log 2>&1 &
# TI
sh run_attack.sh 4 3 10 1.0 0 True   False ./rand_6.txt ours_rand6_TI7_MI_FGSM 1> ./out/ours_rand6_TI7_MI_FGSM.log 2>&1 &
sh run_attack.sh 5 3 10 1.0 0 True   False ./rand_7.txt ours_rand7_TI7_MI_FGSM 1> ./out/ours_rand7_TI7_MI_FGSM.log 2>&1 &
sh run_attack.sh 6 3 10 1.0 0 True   False ./rand_8.txt ours_rand8_TI7_MI_FGSM 1> ./out/ours_rand8_TI7_MI_FGSM.log 2>&1 &
sh run_attack.sh 7 3 10 1.0 0 True   False ./rand_9.txt ours_rand9_TI7_MI_FGSM 1> ./out/ours_rand9_TI7_MI_FGSM.log 2>&1 &
sh run_attack.sh 8 3 10 1.0 0 True   False ./rand_10.txt ours_rand10_TI7_MI_FGSM 1> ./out/ours_rand10_TI7_MI_FGSM.log 2>&1 &
#sh run_attack.sh 7 3 10 1.0 0 True   False ./autoaug_op5_avg.txt ours_op5_avg_TI7_MI_FGSM 1> ./out/ours_op5_avg_TI7_MI_FGSM.log 2>&1 &
#sh run_attack.sh 7 10 10 1.0 0.5 False  False None DI_05_MI_FGSM 1> ./out/DI_05_MI_FGSM.log 2>&1 &
#sh run_attack.sh 7 10 10 1.0 0.5 True   False None DI_05_TI7_MI_FGSM 1> ./out/DI_05_TI7_MI_FGSM.log 2>&1 &
wait
echo "Done."