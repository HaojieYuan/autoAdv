# CUDA_ID 8
# ATTACK TI, TI_DIM_attack_out, SI_TI_MI_FGSM, ours_TI_MIFGSM

bash -i eval_attack_on_defenses.sh 8 "TI"  1> ./defense_eval_result/ti_mi_fgsm_log.log 2>&1
bash -i eval_attack_on_defenses.sh 8 "TI_DIM_attack_out" 1> ./defense_eval_result/di_ti_mi_fgsm_log.log 2>&1
bash -i eval_attack_on_defenses.sh 8 "SI_TI_MI_FGSM" 1> ./defense_eval_result/si_ti_mi_fgsm_log.log 2>&1
#sh eval_attack_on_defenses.sh 8 "ours_TI_MIFGSM"