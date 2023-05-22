checkpoints=(/data/hyxu/codes/nmt-corrector/chpk-nmt/ur-en-chpk/retrain-test_encoder/checkpoint-2000 \
            )

eval_path=/data/hyxu/codes/nmt-corrector/src/eval.sh

echo "环境为hyxu_cor"
for c in ${checkpoints[@]}
do
    bash $eval_path $c
    echo "evaluate ${c} over"
done

echo "evaluate over 。。。。"

# nohup bash ./scripts/eval_ur-en.sh > ./log/eval1.log 2>&1 &