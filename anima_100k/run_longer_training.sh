

set -x -e

run_id=$(date +%s)
echo "RUN ID: $run_ts"

echo "START TIME: $(date)"


ROOT_DIR_BASE=/home/ubuntu/Anima_run
OUTPUT_PATH=$ROOT_DIR_BASE/output_$run_id

mkdir -p $OUTPUT_PATH





python longer_training.py --dataset="DATASET_PATH" \
    --dataset_format="long_data" \
    --learning_rate 0.0001 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1000 \
    --model_name_or_path "lyogavin/Anima-7B-100K" `# base model ` \
    --source_max_len 92000  `# max input len set to input+output ~= 100k  `\
    --target_max_len 1024 `# max output len set to input+output ~= 100k `\
    --eval_dataset_size 1 `# mainly for testing, no need to be big` \
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 10 `# 2 for debug mode only, 10 for training`  \
    --lora_r 32 \
    --bits 16 \
    --bf16 \
    --optim "paged_adamw_8bit" `# 8bit adam to further save mem in optimizer states` \
    --output_dir $OUTPUT_PATH \
    --report_to 'wandb' \
    --sample_generate `# test sample generation every once a while`  \
    --save_steps 10 `# 4 for debug mode only, 10 for training` \
    --trust_remote_code `# use remote code in the hf repo`
    #--training_memory_tracking `turn on for debug oom` \
    #--debug_mode `# only set when it's debug mode` \
