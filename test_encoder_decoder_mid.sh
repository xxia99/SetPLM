#!/bin/bash
home_dir="/home/xiaxin/setplm"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=2

#data_dir="data/kp20k_10k"

#seed=27
#dropout=0.1
#learning_rate=0.0001
#batch_size=48
#copy_attention=true
#epochs=100
#early_stop_tolerance=4
#checkpoint_interval=4000
#max_kp_len=6
#max_kp_num=20
#loss_scale_pre=0.2
#loss_scale_ab=0.1
#set_loss=true
#assign_steps=2
plm="t5-base"
#model_name="One2set"
#data_args="Mid"
##main_args="Seed${seed}_Dropout${dropout}_LR${learning_rate}_BS${batch_size}_MaxLen${max_kp_len}\
##_MaxNum${max_kp_num}_LossScalePre${loss_scale_pre}_LossScaleAb${loss_scale_ab}_Step${assign_steps}_EarStop${early_stop_tolerance}_Epochs${epochs}"
#main_args="PLM${plm}_BS${batch_size}_MaxLen${max_kp_len}\
#_MaxNum${max_kp_num}_LossScalePre${loss_scale_pre}_LossScaleAb${loss_scale_ab}_Step${assign_steps}_EarStop${early_stop_tolerance}_Epochs${epochs}"

#if [ ${copy_attention} = true ] ; then
#    model_name+="_Copy"
#fi
#if [ "${set_loss}" = true ] ; then
#    main_args+="_SetLoss"
#fi
#
#save_data="${data_dir}/${plm}/${data_args}"
#mkdir -p ${save_data}
#
#exp="${data_args}_${model_name}_${main_args}"

#echo "============================= preprocess: ${save_data} ================================="
#
#preprocess_out_dir="output/preprocess/plm/${plm}/${data_args}"
#mkdir -p ${preprocess_out_dir}
#
#cmd="python3 preprocess.py \
#-data_dir=${data_dir} \
#-save_data_dir=${save_data} \
#-remove_title_eos \
#-log_path=${preprocess_out_dir} \
#-plm ${plm} \
#-one2many
#"
#
#echo $cmd
#eval $cmd


#echo "============================= train: ${exp} ================================="
#
#train_out_dir="output/train/plm/encoder_decoder/${plm}/${exp}/"
#mkdir -p ${train_out_dir}
#
#cmd="python3 train.py \
#-data ${save_data} \
#-vocab ${save_data} \
#-exp_path ${train_out_dir} \
#-model_path=${train_out_dir} \
#-learning_rate ${learning_rate} \
#-one2many \
#-batch_size ${batch_size} \
#-seed ${seed} \
#-dropout ${dropout} \
#-fix_kp_num_len \
#-max_kp_len ${max_kp_len} \
#-max_kp_num ${max_kp_num} \
#-loss_scale_pre ${loss_scale_pre} \
#-loss_scale_ab ${loss_scale_ab} \
#-assign_steps ${assign_steps} \
#-seperate_pre_ab \
#-epochs ${epochs} \
#-early_stop_tolerance ${early_stop_tolerance} \
#-plm ${plm} \
#-checkpoint_interval ${checkpoint_interval} \
#-decoder ${plm} \
#-encoder ${plm} \
#"
#
#if [ "${copy_attention}" = true ] ; then
#    cmd+=" -copy_attention"
#fi
#if [ "${set_loss}" = true ] ; then
#    cmd+=" -set_loss"
#fi
#
#echo $cmd
#eval $cmd
#
#echo "============================= test: ${exp} ================================="
#
for data in "kp20k"
##for data in "inspec" "krapivin" "nus" "semeval" "kp20k"
do
#  echo "============================= testing ${data} ================================="
  test_out_dir="output/test/plm/encoder_decoder/${plm}/Mid_One2set_PLMt5-base_BS12_MaxLen6_MaxNum20_LossScalePre0.2_LossScaleAb0.1_Step2_EarStop4_Epochs100/kp20k"
  #  mkdir -p ${test_out_dir}

  src_file="data/testsets/${data}/test_src.txt"
  trg_file="data/testsets/${data}/test_trg.txt"
  kws_file="data/testsets/${data}/test_kws.txt"
  #
  #  cmd="python3 predict.py \
  #  -vocab ${save_data} \
  #  -src_file=${src_file} \
  #  -kws_file=${kws_file} \
  #  -pred_path ${test_out_dir} \
  #  -exp_path ${test_out_dir} \
  #  -model ${train_out_dir}/best_model.pt \
  #  -remove_title_eos \
  #  -batch_size 20 \
  #  -replace_unk \
  #  -dropout ${dropout} \
  #  -fix_kp_num_len \
  #  -one2many \
  #  -vocab_size 50000 \
  #  -max_kp_len ${max_kp_len} \
  #  -max_kp_num ${max_kp_num} \
  #  -seperate_pre_ab \
  #  -plm ${plm} \
  #  -decoder ${plm} \
  #  -encoder ${plm} \
  #  "
  #  if [ "$copy_attention" = true ] ; then
  #      cmd+=" -copy_attention"
  #  fi
  #
  #  echo $cmd
  #  eval $cmd

  cmd="python3 evaluate_prediction.py \
  -pred_file_path ${test_out_dir}/new_prediction.txt \
  -src_file_path ${src_file} \
  -trg_file_path ${trg_file} \
  -exp_path ${test_out_dir} \
  -kws_file_path ${kws_file} \
  -export_filtered_pred \
  -filtered_pred_path ${test_out_dir} \
  -disable_extra_one_word_filter \
  -invalidate_unk \
  -all_ks 5 M \
  -present_ks 5 M \
  -absent_ks 5 M
  ;cat ${test_out_dir}/results_log_5_M_5_M_5_M.txt
  "

  echo $cmd
  eval $cmd

done