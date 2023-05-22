
set -e

echo "环境为env"

checkpoint=$1
name=$2
src_prefix=$3
tgt_prefix=$4
src_lang=$5
tgt_lang=$6
src_lang_code=$7
tgt_lang_code=$8

CUDA_VISIBLE_DEVICES="0"  python3 ./src/nmt_retrain.py \
    --source_lang ${src_lang} \
    --target_lang ${tgt_lang} \
    --src_lang_code ${src_lang_code} \
    --tgt_lang_code ${tgt_lang_code} \
    --saved_dir 'chpk-nmt/' \
    --checkpoint ${checkpoint} \
    --num_beams 0 \
    --lr 4e-6 \
    --warmup_steps 200  \
    --batch_size 16 \
    --eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --max_sentence_length 128   \
    --max_generate_length 128   \
    --data_dir 'data/en-ur/' \
    --src_file "${src_prefix}.en-ur.${src_lang},${src_lang}"   \
    --tgt_file "${tgt_prefix}.en-ur.${tgt_lang},${tgt_lang}"   \
    --bleu_tokenize ''  \
    --evaluate_metrics 'bleu'\
    --early_stopping_patience 4 \
    --eval_steps 1000    \
    --save_steps 1000    \
    --logging_steps 100 \
    --test_dataset opus  \
    --freeze_decoder true \
    --name ${name}  \
    --des ''    \
           
# ur_PK, en_XX
# sl_SI, en_XX
# /public/home/hongy/pre-train_model/opus-mt-ur-en
# /public/home/hongy/hyxu/nmt-corrector/checkpoint-nmt/ur-en-flores/checkpoint-4000 04-07-23:
# nohup ./src/nmt_retrain.sh > ./log/retrain-ur-en-10w.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/train_mbart50.log -gpu  num=1:mode=exclusive_process sh ./src/nmt_train.sh