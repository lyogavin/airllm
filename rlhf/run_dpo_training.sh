

set -x -e

run_id=$(date +%s)
echo "RUN ID: $run_ts"

echo "START TIME: $(date)"


ROOT_DIR_BASE=./Anima_run
OUTPUT_PATH=$ROOT_DIR_BASE/output_$run_id

mkdir -p $OUTPUT_PATH






python qlora_dpo.py --dataset="lyogavin/Anima33B_rlhf_belle_eval_1k" `# rlhf dataset` \
    --dataset_format="hh-rlhf" `# follow hh-rlhf format` \
    --learning_rate 0.0001 `# QLoRA paper appendix B Table 9 `\
    --per_device_train_batch_size 1 `# fix for fitting mem `\
    --gradient_accumulation_steps 16 `# QLoRA paper appendix B Table 9  `\
    --max_steps 100 `# run 100 steps`\
    --model_name_or_path "lyogavin/Anima33B-merged" `# the base model to train on` \
    --reference_model "lyogavin/Anima33B-merged" `# the reference model the training should reference` \
    --source_max_len 600  `# 600 rougly covers 90PT of lengths`\
    --target_max_len 600 `# 600 rougly covers 90PT of lengths`\
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 10 `# eval every 10 steps to make sure we monitor the whole training process`  \
    --output_dir $OUTPUT_PATH \
    --report_to 'wandb' \
    --sample_generate `# test sample generation every once a while`  \
    --save_steps 10 `# save every 10 steps to make sure we can reproduce the whole training process` \
    --train_on_source true \
    --lora_r 256 \
    --beta 0.1 `# Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.`
    #--debug_mode `# only set when it's debug mode` \
