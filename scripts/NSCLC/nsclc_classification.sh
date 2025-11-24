#!/bin/bash

gpuid=$1
config=$2

### Dataset Information
declare -a dataroots=(
	'data/TCGA/NSCLC_both/extracted_mag20x_patch256_fp/feats_pt'
)

task='NSCLC'
target_col='strat_label'
split_dir="classification/NSCLC_filter"
split_names='train,val,test'

bash "./scripts/classification/NSCLC/${config}.sh" $gpuid $task $target_col $split_dir $split_names $dataroots 
