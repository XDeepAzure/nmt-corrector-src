
import os
import torch
from tqdm import tqdm
from data_process import compute_metric
import evaluate
from utils import create_logger, avg, compute_bleu, compute_bleurt, compute_chrf, compute_ter, compute_batch
import fire

metric = evaluate.load("sacrebleu")
def compute_batch(ref, pre, metric_name="bleu"):
    """暂时只能默认的使用metric_name = bleu的"""
    bleu = []
    for r, p in tqdm(list(zip(ref, pre))):
        bleu.append(compute_metric(r, p, metric)['score'])
    return bleu

def main(pairs = "ka-en"):

    logger = create_logger(f"/public/home/hongy/hyxu/nmt-corrector/log/chatgpt_{pairs}_opus.log")

    path = f"/public/home/hongy/hyxu/nmt-corrector/data/src-ref-pre/{pairs}-opus-test"

    results_pre, results_cor = [], []
    data, data_filte, bleu = [], [], []
    for file_path in os.listdir(path):
        if not file_path.startswith("thread"): continue
    
        id_thread = int(file_path[len("thread"):])
        data_path = os.path.join(path, file_path, f"paras-{id_thread}.bin")
        data = torch.load(data_path)
    
        bleu += compute_batch(ref=[p.ref for p in data],
                         pre=[p.pre for p in data])
    
        data_filte += [p for p, b in zip(data, bleu) if b>0]
    ref = [[p.ref] for p in data_filte]
    pre = [p.pre for p in data_filte]
    bleurt = compute_bleurt(predictions=pre, references=ref)["score"]
    chrf = compute_chrf(predictions=pre, references=ref)["score"]
    ter = compute_ter(predictions=pre, references=ref)["score"]

    logger.info(f"bleu: {avg([b for b in bleu if b>0])}")
    logger.info(f"bleurt: {bleurt}")
    logger.info(f"chrf++: {chrf}")
    logger.info(f"ter: {ter}")

if __name__ == "__main__":
    fire.Fire(main)