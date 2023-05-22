
retrain_path=/data/hyxu/codes/nmt-corrector/src/nmt_retrain.sh

checkpoint=/data/hyxu/codes/nmt-corrector/chpk-nmt/ur-en/opus-dev/checkpoint-24000


bash ${retrain_path} ${checkpoint} test 30w pre30w #添加前缀

# bash ${retrain_path} ${checkpoint} 8w-mono20-b4 cor20w.mono

# nohup bash ./scripts/retrain.sh > ./log/train.log 2>&1 &