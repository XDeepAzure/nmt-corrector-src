from typing import List
import torch
import os
import evaluate
from tqdm import tqdm

from utils import SrcRefPreCor, create_logger, paras_filter_by_belu, compute_batch, avg

# nohup python ./src/data_process.py > run.log 2>&1 &

logger = create_logger("./log/process_data_train_nmt.log")
metric = evaluate.load('sacrebleu')

def filter_fn(paras, r_p_bleu, r_c_bleu, return_bleu=False):
    """返回过滤后的SrcRefPreCor对,和对应的bleu, r_p_bleu在第二位， r_c_bleu在第三位"""
    data = [(paras[i], p_b, c_b) for i, (p_b, c_b) in enumerate(zip(r_p_bleu, r_c_bleu)) if c_b>5 and c_b>p_b-5]
    if return_bleu:
        return data
    else:
        return [p for p,pb,cb in data]

def compute_metric(ref, pre, metric=None):
    if metric==None:
        metric = evaluate.load('sacrebleu')
    # bleu = metric.compute(predictions=[pre], references=[[ref]], tokenize="flores200")
    bleu = metric.compute(predictions=[pre], references=[[ref]])
    return bleu

def compute_batch(ref, pre, metric_name="bleu"):
    """暂时只能默认的使用metric_name = bleu的"""
    bleu = []
    for r, p in tqdm(list(zip(ref, pre))):
        bleu.append(compute_metric(r, p, metric)['score'])
    return bleu

def clear_fn(paras: List[SrcRefPreCor], src_ref_pre_cor: List[SrcRefPreCor], is_filter=True):
    """清洗得到的cor，去掉两端多余回车和两端空格，去掉cor为空的数据
    计算ref与pre之间的bleu和ref与cor之间的bleu并返回，不过滤的话就不返回过滤的bleu
    
    Args:
        paras (List[SrcRefPreCor]): 需要处理的数据
        src_ref_pre_cor (List[SrcRefPreCor]): 将过滤后的数据存在此处
        is_filter (bool, optional): Defaults to True.是否过滤，过滤策略是filter_fn决定的

    Returns:
        _type_: bleu 如果过滤就还返回过滤的bleu
    """

    ref = [p.ref for p in paras if p.cor != None]
    pre = [p.pre for p in paras if p.cor != None]
    cor = [p.cor.replace("\n", " ").lstrip(" ").rstrip(" ") for p in paras if p.cor!=None]
    r_p_bleu = compute_batch(ref=ref, pre=pre)
    r_c_bleu = compute_batch(ref=ref, pre=cor)
    # src_ref_pre_cor += [paras[i] for i, (p_b, c_b) in enumerate(zip(r_p_bleu, r_c_bleu)) if c_b>p_b]
    
    paras = [p for p in paras if p.cor!=None]
    for p, c in zip(paras, cor):                    #将干净的cor赋值给p
        p.cor = c
    
    if is_filter:
        data = filter_fn(paras, r_p_bleu, r_c_bleu, return_bleu=True)
        src_ref_pre_cor += [p for p,pb,cb in data]
        filter_p_bleu = [pb for p,pb,cb in data]
        filter_c_bleu = [cb for p,pb,cb in data]
        return r_p_bleu, r_c_bleu, filter_p_bleu, filter_c_bleu
    else:
        src_ref_pre_cor += paras
        return r_p_bleu, r_c_bleu

def process_data_from_dir(data_dir, is_filter=True, return_bleu=False):
    """data_dir 进程结果保存位置，clear_fn 清理函数"""
    file_prefix = 'thread'
    paras_prfix = 'paras-'
    total_r_p_bleu, total_r_c_bleu, filter_r_p_bleu, filter_r_c_bleu = [], [], [], [] 
    total_paras = []
    src_ref_pre_cor = []
    dir_list = os.listdir(data_dir)
    for i, p in enumerate(dir_list):
        if not p.startswith(file_prefix): continue
    
        id_thread = p[len(file_prefix):]
        logger.info(f"正在处理第{id_thread}个进程结果")
        paras_data = torch.load(os.path.join(data_dir, p, f"{paras_prfix}{id_thread}.bin"))
        total_paras += [p for p in paras_data if p.cor != None]
    
        result = clear_fn(paras_data, src_ref_pre_cor, is_filter)
        if is_filter:
            r_p_bleu, r_c_bleu, filter_p_bleu, filter_c_bleu = result
            filter_r_c_bleu += filter_c_bleu
            filter_r_p_bleu += filter_p_bleu
            logger.info(f"过滤后的pre平均bleu:{avg(filter_p_bleu)},  cor平均bleu:{avg(filter_c_bleu)}")
        else:
            r_p_bleu, r_c_bleu = result
        logger.info(f"pre平均bleu:{avg(r_p_bleu)},  cor平均bleu:{avg(r_c_bleu)}")
        total_r_p_bleu += r_p_bleu
        total_r_c_bleu += r_c_bleu
    logger.info(f"全部进程的总pre平均bleu:{avg(total_r_p_bleu)},  cor平均bleu:{avg(total_r_c_bleu)}")
    if return_bleu:
        return src_ref_pre_cor, total_paras, (total_r_p_bleu, total_r_c_bleu), (r_p_bleu, r_c_bleu)
    else:
        return src_ref_pre_cor, total_paras
if __name__ == "__main__":
    pairs = "sl-en"
    data_path = f"/public/home/hongy/hyxu/nmt-corrector/data/src-ref-pre/{pairs}-all/7-50"
    save_path = f"/public/home/hongy/hyxu/nmt-corrector/data/{pairs}/process/"
    src_ref_pre_cor , total_paras, total_bleu, filter_bleu = process_data_from_dir(data_path, True, True)
    
    torch.save(total_paras, os.path.join(data_path, "total_paras.bin"))
    torch.save(src_ref_pre_cor, os.path.join(data_path, "fliter_paras.bin"))
    torch.save(total_bleu, os.path.join(data_path, "bleu.bin"))
    logger.critical(f"过滤后得到的数据的总量为{len(src_ref_pre_cor)}")
    # tan_ur_data = []
    # tan_en_data = []
    # with open(os.path.join(save_path, "Tanzil.en-ur.en"), 'r') as e_f , open(os.path.join(save_path, "Tanzil.en-ur.ur"), "r") as u_f:
    #     tan_en_data = e_f.readlines()
    #     tan_ur_data = u_f.readlines()

    cor_data = [p.cor+'\n' for p in src_ref_pre_cor]
    cor_src_data = [p.src+"\n" for p in src_ref_pre_cor]
    cor_pre_data = [p.pre+"\n" for p in src_ref_pre_cor]

    with open(os.path.join(save_path, f"24w.train.{pairs}.sl"), "w") as src_f,  \
        open(os.path.join(save_path, f"24w.train.{pairs}.en"), 'w') as cor_f, \
        open(os.path.join(save_path, f"24w.pre.{pairs}.en"), 'w') as pre_f: 
        src_f.writelines(cor_src_data)
        cor_f.writelines(cor_data)
        pre_f.writelines(cor_pre_data)
    
    # logger.critical(f"开始调用nmt的训练脚本")
    # os.system("./nmt_train.sh")