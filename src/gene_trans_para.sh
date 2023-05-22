# 生成generate correct paras的脚本

echo "环境为env"

# CUDA_VISIBLE_DEVICES="6" 
python3 ./src/gene_trans_para.py \
    --source_lang sl \
    --target_lang en \
    --src_lang_code "sl_SI" \
    --tgt_lang_code "en_XX" \
    --checkpoint "/public/home/hongy/hyxu/nmt-corrector/checkpoint-nmt/sl-en-chpk/sl-en/checkpoint-120000"  \
    --saved_dir 'data/src-ref-pre' \
    --batch_size 80 \
    --max_sentence_length 128   \
    --max_length 128   \
    --data_dir 'data/en-sl'\
    --src_file 'filter.en-sl.sl'\
    --tgt_file 'filter.en-sl.en'\
    --num_beams 0   \
    --num_sentence 0 \
    --src_mono_file true    \

    # --data_dir '/public/home/hongy/hyxu/nmt-corrector/data/correct_pairs/ur-en-5000'\
    #urd_Arab, eng_Latn, tur_Latn, slv_Latn, kat_Geor, azb_Arab
# nohup bash ./src/gene_trans_para.sh > ./log/mbart-gene-sl-en.log 2>&1 &
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o ./log/gene_mbart50.log -gpu  num=1:mode=exclusive_process sh ./src/gene_correct_p.sh