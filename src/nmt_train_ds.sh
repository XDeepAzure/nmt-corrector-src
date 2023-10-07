hostfile=""
deepspeed --hostfile=$hostfile src/nmt_train_ds.py  \
    --report_to "tensorboard" \
    --src_file "/apdcephfs_cq2/share_1567347/hayuxu/data/nmt/zh-en/ELRC_2922/en-zh.zh" \
    --tgt_file "/apdcephfs_cq2/share_1567347/hayuxu/data/nmt/zh-en/ELRC_2922/en-zh.en" \
    --model_name_or_path "/apdcephfs_cq2/share_1567347/hayuxu/models/mt5-small" \
    --output_dir "/apdcephfs_cq2/share_1567347/hayuxu/models" \
    --model_max_length 256 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --logging_steps 4 \
    --gradient_checkpointing False \
    --deepspeed src/ds_config.json \
    --fp16 True \
    --fp16_opt_level O2\