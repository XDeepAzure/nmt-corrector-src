import multiprocessing
from happytransformer import HappyTextToText, TTSettings
import os
import torch
from tqdm import tqdm
from logging import getLogger

logger = getLogger()

DEBUG = False

def T5_base_cor(pre_data):
    from happytransformer import HappyTextToText, TTSettings
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

    args = TTSettings(num_beams=5, min_length=1)

    total_result = []
    for s in tqdm(pre_data):
        res = happy_tt.generate_text(f"grammar: {s}", args=args)
        total_result.append(res.text)
    return total_result

def T5_large_cor(pre_data):
    total_result = []
    from transformers import pipeline
    corrector = pipeline(
        "text2text-generation",
        "pszemraj/flan-t5-large-grammar-synthesis",
        device=torch.device("cuda:0")
    )
    for s in tqdm(pre_data):
        res = corrector(s)
        total_result.append(res[0]["generated_text"])
    return total_result

def thread_fun(id_thread, pre_data, path):

    if id_thread < 5:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # total_result = T5_base_cor(pre_data)
    total_result = T5_large_cor(pre_data)
    total_result = [s+"\n" for s in total_result]
    dir_path = os.path.join(path, f"thread{id_thread}")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    with open(os.path.join(dir_path, "res.txt"), 'w') as f:
        f.writelines(total_result)
    print("存入完成")

def process_result(path):
    dir_lst = os.listdir(path)
    results = []
    for d in dir_lst:
        if "thread" not in d: continue
        res_path = os.path.join(path, d, "res.txt")
        with open(res_path, 'r') as f:
            res = f.readlines()
        results += res
    with open(os.path.join(path, "30w-t50-en-ur.en"), 'w') as f:
        f.writelines(results)



if __name__ == '__main__':
    data_path = "./data"
    src_ref_pre_path = os.path.join(data_path, "pre30w-en-ur.en")

    logger.info(src_ref_pre_path)

    with open(src_ref_pre_path, 'r') as f:
        src_ref_pre = f.readlines()

    src_ref_pre_filt = src_ref_pre

    ## ! debug
    if DEBUG:
        src_ref_pre_filt = src_ref_pre_filt[:9]

    ## 设置num_thread
    num_thread = 8 if not DEBUG else 2
    num_sent_pre_thread = len(src_ref_pre_filt) // num_thread
    # 将句子列表拆分，给每个子线程一个句子列表， 最后一个进程要把剩下的全部包括进去
    thread_src_ref_pre = [src_ref_pre_filt[i*num_sent_pre_thread : (i+1)*num_sent_pre_thread] \
                      if i!=num_thread-1 else src_ref_pre_filt[i*num_sent_pre_thread : ]   \
                      for i in range(0, num_thread) ]

    logger.info(f"every thread process num of sentence {num_sent_pre_thread}, the last thread process num {len(thread_src_ref_pre[-1])}")
    pool = multiprocessing.Pool(processes=num_thread)
    processes = []
    for i in range(num_thread):
        p = pool.apply_async(thread_fun, (i, thread_src_ref_pre[i], data_path))
        processes.append(p)
    pool.close()
    pool.join()
    for p in processes:
        p.get()
    print("------------处理结果-------------")
    process_result(data_path)
    print("------------处理完成-------------")