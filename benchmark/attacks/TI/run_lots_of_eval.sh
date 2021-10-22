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
#sh run_attack.sh 8 10 10 1.0 0 True  False ./autoaug_b1.txt ours_TI_MI_FGSM_b1_ti15 1> ./out/ours_TI_MI_FGSM_b1_ti15_log.log 2>&1

#sh run_attack.sh 8 3 10 1.0 0 False False ./autoaug_b3.txt ours_MI_FGSM_b3_ti7 1> ./out/ours_MI_FGSM_b3_ti7_log.log 2>&1
#sh run_attack.sh 8 3 10 1.0 0 True  False ./autoaug_b3.txt ours_TI_MI_FGSM_b3_ti15 1> ./out/ours_TI_MI_FGSM_b3_ti15_log.log 2>&1

#sh run_attack.sh 8 1 10 1.0 0 False False ./autoaug_b5.txt ours_MI_FGSM_b5_ti7 1> ./out/ours_MI_FGSM_b5_ti7_log.log 2>&1
#sh run_attack.sh 8 1 10 1.0 0 True  False ./autoaug_b5.txt ours_TI_MI_FGSM_b5_ti15 1> ./out/ours_TI_MI_FGSM_b5_ti15_log.log 2>&1

#sh run_attack.sh 8 1 10 1.0 0 False False ./autoaug_b7.txt ours_MI_FGSM_b7_ti7 1> ./out/ours_MI_FGSM_b7_ti7_log.log 2>&1
#sh run_attack.sh 8 1 10 1.0 0 True  False ./autoaug_b7.txt ours_TI_MI_FGSM_b7_ti15 1> ./out/ours_TI_MI_FGSM_b7_ti15_log.log 2>&1

#sh run_attack.sh 8 1 10 1.0 0 False False ./autoaug_b9.txt ours_MI_FGSM_b9_ti7 1> ./out/ours_MI_FGSM_b9_ti7_log.log 2>&1
#sh run_attack.sh 8 1 10 1.0 0 True  False ./autoaug_b9.txt ours_TI_MI_FGSM_b9_ti15 1> ./out/ours_TI_MI_FGSM_b9_ti15_log.log 2>&1
#sh run_attack.sh 1 2 10 1.0 0 False False ./autoaug_b5.txt IJCAI_ours 1> ./out/IJCAI_ours.log 2>&1 &
#sh run_attack.sh 0 2 10 1.0 0 False False ./autoaug_b5_2.txt IJCAI_ours2 1> ./out/IJCAI_ours2.log 2>&1 &
#                       TI      k   SI      DEM     NI      ours    out

#sh run_attack_v3.sh 7 4 False   7   True    False   False   None    GHOSTNET_5  1>./out/GHOSTNET_5.log 2>&1 &
#sh run_attack_v3.sh 8 4 False   7   True    False   True    None    NI_GHOSTNET_5  1>./out/NI_GHOSTNET_5.log 2>&1 &
#sh run_attack_v3.sh 9 4 True    15  True    False   False   None    TI_GHOSTNET_5  1>./out/TI_GHOSTNET_5.log 2>&1 &

#wait
#exit 0
#sh run_attack_v3.sh 8 4 False   7   False   False   True    None    NI_GHOSTNET  1>./out/NI_GHOSTNET.log 2>&1 &

#echo GHOST
#sh run_attack_v3.sh 5 4 False    15  False   False   False   None    GHOST 1>./out/GHOST.log 2>&1


sh run_attack.sh  0 4    True   15   False   False   False   ./autoaug_b5_1.txt    ours  1>./out/ours.log 2>&1 &
#sh run_attack.sh 1 4    True    7   False   False   False   ./autoaug_b5_1.txt    IJCAI_ours_TI7 1>./out/IJCAI_ours_TI7.log 2>&1 &
#sh run_attack.sh 3 4    False   7   False   False   True    ./autoaug_b5_1.txt    IJCAI_ours_NI 1>./out/IJCAI_ours_NI.log 2>&1 &
#wait

#sh run_attack.sh 0 4    True    7   False   False   False   ./random_1.txt        IJCAI_random1  1>./out/IJCAI_random1.log 2>&1 &
#sh run_attack.sh 1 4    True    7   False   False   False   ./random_2.txt        IJCAI_random2  1>./out/IJCAI_random2.log 2>&1 &
#sh run_attack.sh 2 4    True    7   False   False   False   ./random_3.txt        IJCAI_random3  1>./out/IJCAI_random3.log 2>&1 &
#sh run_attack.sh 3 4    True    7   False   False   False   ./random_4.txt        IJCAI_random4  1>./out/IJCAI_random4.log 2>&1 &
#sh run_attack.sh 4 4    True    7   False   False   False   ./random_5.txt        IJCAI_random5  1>./out/IJCAI_random5.log 2>&1 &
#sh run_attack.sh 3 4    True    7   False   False   False   ./random_4.txt        clean  1>./out/clean.log 2>&1 &
#sh run_attack.sh 3 4    True    15   False   False   False   ./1_branch.txt        1_branch  1>./out/1_branch.log 2>&1 &
#sh run_attack.sh 1 4    True    15   False   False   False   ./3_branch.txt        3_branch  1>./out/3_branch.log 2>&1 &
#sh run_attack.sh 2 2    True    15   False   False   False   ./7_branch.txt        7_branch  1>./out/7_branch.log 2>&1 &
#sh run_attack.sh 2 2    False    15   False   False   False   ./fix_on_AlexNet.txt       fix_on_AlexNet_MIM  1>./out/fix_on_AlexNet_MIM.log 2>&1 &
#sh run_attack.sh 3 2    False    15   False   False   False   ./fix_on_MnasNet.txt       fix_on_MnasNet_MIM  1>./out/fix_on_MnasNet_MIM.log 2>&1 &
