
echo "23-05-15晚上的任务"

# CUDA_VISIBLE_DEVICES="0" 

path="/data/hyxu/codes/nmt-corrector/src/nmt_retrain.sh"

echo "----------------------start--------------------------"
checkpoint=/data/hyxu/codes/nmt-corrector/chpk-nmt/ur-en-chpk/opus-dev/checkpoint-24000
bash $path $checkpoint test 30w 30w-t5-large ur en ur_PK en_XX 

# echo "----------------------start--------------------------"
# checkpoint=/public/home/hongy/hyxu/nmt-corrector/checkpoint-nmt/sl-en-chpk/sl-en/checkpoint-120000
# bash $path $checkpoint 25w-old16 25w.train 25w.train sl en sl_SI en_XX 


# nohup ./scripts/task.sh > ./log/retrain.log 2>&1 &