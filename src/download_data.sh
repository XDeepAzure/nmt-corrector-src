# 下载opus-100和flores-100语料库中的某几对语言的测试和开发集

saved_dir="/SISDC_GPFS/Home_SE/hy-suda/hyxu/nmt-corrector/data/en-ur"

lianjies=('https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ur/opus.en-ur-test.en' \
          'https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ur/opus.en-ur-test.ur' \
          'https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ur/opus.en-ur-dev.ur' \
          'https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ur/opus.en-ur-dev.en' \
          )

for v in ${lianjies[@]}
do
    wget --no-check-certificate -p $saved_dir $v
    echo "download over from {$v}"
done
