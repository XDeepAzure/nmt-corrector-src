# /bin/bash

echo "🌹 start ... "

echo "这个是base环境"
# 因为在continue中不知道为啥env的path变量中没有添加pip的路径，所以这里要用python -m pip 来安装
python -m pip install gpustat

## ! 后续可以尝试将pip的路径添加到path中去

alias kkgpu="watch --color -n1 gpustat -cpu --color"

source ~/anaconda3/bin/activate work

conda activate work

echo "这个是conda启动的work环境"

python -m ensurepip

python -m pip install tensorboard peft transformers_stream_generator datasets
python -m pip install pynvml openpyxl scipy evaluate matplotlib
#pip install bitsandbytes
# python3 gpu_liyong.py
alias kkgpu="watch --color -n1 gpustat -cpu --color"

# python3 gpu.py
# sleep 100d
echo "🌈 begin train ...."

# bash train.sh
bash src/nmt_train.sh

echo "🐯 train end ...."

# sleep 200d