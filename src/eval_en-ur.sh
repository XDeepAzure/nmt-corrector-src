checkpoints=(/public/home/hongy/hyxu/nmt-corrector/checkpoint-nmt/en-ur-chpk-bug/en-ur/checkpoint-88000 \
            /public/home/hongy/hyxu/nmt-corrector/checkpoint-nmt/en-ur-chpk-bug/en-ur-100w/checkpoint-84000 \
            /public/home/hongy/hyxu/nmt-corrector/checkpoint-nmt/en-ur-chpk-bug/en-ur-pre100w/checkpoint-46000 \
            /public/home/hongy/hyxu/nmt-corrector/checkpoint-nmt/en-ur-chpk-bug/en-ur-retrain-30w/checkpoint-1000 \
            /public/home/hongy/hyxu/nmt-corrector/checkpoint-nmt/en-ur-chpk-bug/en-ur-retrain-30w_decoder/checkpoint-1000 \
            /public/home/hongy/hyxu/nmt-corrector/checkpoint-nmt/en-ur-chpk-bug/en-ur-retrain-30w_encoder/checkpoint-1000 \
            /public/home/hongy/hyxu/nmt-corrector/checkpoint-nmt/en-ur-chpk-bug/en-ur-retrain-30wpre_decoder/checkpoint-1000 \
)

echo "环境为hyxu_env"
for c in ${checkpoints[@]}
do
    bash /public/home/hongy/hyxu/nmt-corrector/src/eval.sh $c
    echo "evaluate ${c} over"
done

echo "evaluate over 。。。。"

# nohup bash ./src/eval1_en-ur.sh > ./log/eval.log 2>&1 &