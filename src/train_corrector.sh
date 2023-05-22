# CUDA_VISIBLE_DEVICES="1"
python3 ./src/train_corrector.py \
    --lang en \
    --lang_code "en_XX" \
    --saved_dir '/SISDC_GPFS/Home_SE/hy-suda/hyxu/nmt-corrector/checkpoint-corrector/mbart50' \
    --lr 2e-5   \
    --batch_size 20 \
    --max_sentence_length 256   \
    --max_generate_length 256   \
    --data_dir '/SISDC_GPFS/Home_SE/hy-suda/hyxu/nmt-corrector/data/correct_pairs'\
    --num_beams 1   \
    --model_name "facebook/mbart-large-50"  \
    --pretrained_model './' \
    --resume_from_checkpoint "" \
    --evaluate_metrics 'bleu,'  \
    --metrics_patience '5,'    \
    --seed 10   \
    --warmup_steps 100  \


# nohup bash ./src/train_corrector.sh > ./log/train_cor_mbart50.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/train_cor_mbart50.log -gpu  num=1:mode=exclusive_process sh ./src/train_corrector.sh