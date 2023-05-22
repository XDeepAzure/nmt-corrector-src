
set -e
# checkpoint=$1

CUDA_VISIBLE_DEVICES="1" python3 ./src/eval.py \
    --source_lang tr \
    --target_lang en \
    --src_lang_code "eng_Latn" \
    --tgt_lang_code "tur_Latn" \
    --checkpoint /data/hyxu/codes/cache_dir/nllb-200-distilled-600M  \
    --saved_dir "/data/hyxu/codes/nmt-corrector/log/nllb" \
    --eval_batch_size 40 \
    --max_sentence_length 256   \
    --max_generate_length 256   \
    --data_dir 'data/en-ur'\
    --src_file eng_Latn \
    --tgt_file tur_Latn \
    --bleu_tokenize 'flores200' \
    --evaluate_metrics 'bleu,chrf,ter,bleurt'  \
    --num_beams 5   \
    --test_dataset flores \


    # --checkpoint ${checkpoint}  \
    # --saved_dir ${checkpoint} \
# checkpoint=$1
#--checkpoint "/public/home/hongy/pre-train_model/nllb-200-distilled-600M"  \
#--data_dir '/public/home/hongy/hyxu/flores'\
# urd_Arab, eng_Latn, tur_Latn, slv_Latn, kat_Geor, mya_Mymr
# nohup bash ./src/eval.sh > ./log/eval1.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/eval.log -gpu  num=1:mode=exclusive_process sh ./src/eval.sh