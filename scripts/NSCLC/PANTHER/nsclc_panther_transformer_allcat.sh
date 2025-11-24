#!/bin/bash

gpuid=$1
task=$2
target_col=$3
split_dir=$4
split_names=$5
dataroots=$6
# proto_path=$7

feat='extracted-vit_large_patch16_224.dinov2.uni_mass100k'
input_dim=1024
mag='20x'
patch_size=256

bag_size='-1'
batch_size=64
out_size=17
out_type='allcat'
model_tuple='PANTHER,default'
lin_emb_model='TransformerEmb'
max_epoch=50
lr=0.0001
wd=0.00001
lr_scheduler='cosine'
opt='adamW'
grad_accum=1
loss_fn='cox'
n_label_bin=4
alpha=0.5
em_step=1
load_proto=1
es_flag=0
tau=1.0
eps=1
n_fc_layer=0
proto_num_samples='1.0e+05'
save_dir_root=results
sample_col='slide_id'
label_col='strat_label'

proto_path=""

IFS=',' read -r model config_suffix <<< "${model_tuple}"
model_config=${model}_${config_suffix}
feat_name=$(echo $feat | sed 's/^extracted-//')
exp_code=${task}/${model_config}/${feat_name}
save_dir=${save_dir_root}/${exp_code}

th=0.00005
if awk "BEGIN {exit !($lr <= $th)}"; then
  warmup=0
  curr_lr_scheduler='constant'
else
  curr_lr_scheduler=$lr_scheduler
  warmup=1
fi

# Identify feature paths
all_feat_dirs="${dataroots}"

# Actual command
cmd="
CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_classification \\
    --data_source "${all_feat_dirs}" \\
    --results_dir "${save_dir}" \\
    --split_dir "${split_dir}" \\
    --split_names "${split_names}" \\
    --task "${task}" \\
    --target_col "${target_col}" \\
    --model_type "${model}" \\
    --model_config "${model}_default" \\
    --n_fc_layers "${n_fc_layer}" \\
    --in_dim ${input_dim} \\
    --opt "${opt}" \\
    --lr "${lr}" \\
    --lr_scheduler "${curr_lr_scheduler}" \\
    --accum_steps "${grad_accum}" \\
    --wd "${wd}" \\
    --warmup_epochs "${warmup}" \\
    --max_epochs ${max_epoch} \\
    --train_bag_size "${bag_size}" \\
    --batch_size "${batch_size}" \\
    --in_dropout 0 \\
    --seed 1 \\
    --num_workers 8 \\
    --em_iter "${em_step}" \\
    --tau ${tau} \\
    --n_proto ${out_size} \\
    --out_type "${out_type}" \\
    --emb_model_type "${lin_emb_model}" \\
    --ot_eps "${eps}" \\
    --out_size ${out_size} \\
    --sample_col ${sample_col} \\
    --label_col ${label_col}"
# Specifiy prototype path if load_proto is True
if [[ $load_proto -eq 1 ]]; then
  cmd="$cmd\\
    --load_proto \\
    --proto_path "${proto_path}" \\
  "
fi

echo "$cmd"
eval "$cmd"