
# CUDA_VISIBLE_DEVICES="4"  
python ./src/nmt_train.py \
    --source_lang zh \
    --target_lang en \
    --src_lang_code zh \
    --tgt_lang_code en \
    --model_name "mt5-small"  \
    --num_beams 0  \
    --saved_dir '/apdcephfs_cq2/share_1567347/hayuxu/models' \
    --resume_from_checkpoint '' \
    --lr 2e-5 \
    --warmup_steps 100  \
    --batch_size 16 \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --max_sentence_length 256   \
    --max_generate_length 256   \
    --src_file 'en-zh.zh'   \
    --tgt_file 'en-zh.en'   \
    --eval_steps 10 \
    --evaluation_strategy "no"\
    --save_steps 10 \
    --evaluate_metrics 'bleu'\
    --bleu_tokenize '' \
    --early_stopping_patience 15 \
    --pretrained_model "/apdcephfs_cq2/share_1567347/hayuxu/models"   \
    --data_dir '/apdcephfs_cq2/share_1567347/hayuxu/data/nmt/zh-en/ELRC_2922'  \
    --test_dataset no  \
    --name 'test'     \

    
    # --src_lang_code "en_XX" \
    # --tgt_lang_code "ur_PK" \
    # --tgt_lang_code "ka_GE" \
    # --src_lang_code "tr_TR "
    # --src_lang_code "sl_SI "
    # --src_lang_code "my_MM "
    # --src_lang_code "be "
    #urd_Arab, eng_Latn, tur_Latn, slv_Latn, kat_Geor, azb_Arab
# nohup ./src/nmt_train.sh > ./log/train-en-sl-total.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/train_mbart50.log -gpu  num=1:mode=exclusive_process sh ./src/nmt_train.sh