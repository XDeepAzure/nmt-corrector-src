"""
生成en-en 
"""

## ! 需要增加生成单语语句的翻译

import torch

import json
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer,
                          MBartForConditionalGeneration,
                          M2M100ForConditionalGeneration
                          )

from utils import(SrcRefPreCor,
                  load_tokenizer,
                  get_translate_paras_from_file,
                  get_mono_from_file,
                  get_tokenized_datasets,
                  get_tokenized_datasets_mono,
                  get_model,
                  initialize_exp,
                  get_data_collator)

from utils import create_logger
from logging import getLogger

logger = getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_lang", type=str
    )
    parser.add_argument(
        "--target_lang", type=str
    )
    parser.add_argument(
        "--src_lang_code", type=str
    )
    parser.add_argument(
        "--tgt_lang_code", type=str
    )
    ## ! 将此步骤完善
    parser.add_argument(
        "--num_sentence", type=int, default=-1, help="要生成的纠正对的数量, -1表示全部"
    )
    parser.add_argument(
        "--saved_dir", type=str, default="/data/correct_pairs"
    )
    parser.add_argument(
        "--seed", type=int, default=10
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="翻译的模型"
    )
    parser.add_argument(
        "--max_sentence_length", type=int, default=256
    )
    parser.add_argument(
        "--max_length", type=int, default=256
    )
    parser.add_argument(
        "--num_beams", type=int, default=1 
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )
    parser.add_argument(
        "--data_dir", type=str, default=""
    )
    parser.add_argument(
        ## TODO 改进代码结构，用着一个源文件也可以单独的进行翻译能力
        "--src_file", type=str, default="", help=""
    )
    parser.add_argument(
        "--tgt_file", type=str, default=""
    )
    parser.add_argument(
        "--src_mono_file", type=lambda x: x=="true", default=False, help="要翻译的是否是单语文本"
    )
    parser.add_argument(
        "--tokenized_datasets", type=str, default=""
    )
    parser.add_argument(
        "--data_collator", type=str, default=""
    )
    args = parser.parse_args()
    return args

def get_dataloader(args, data_collator, tokenizer=None, split='train', num_sentence=-1):
    # if args.tokenized_datasets == "":
    #     tokenized_datasets = get_tokenized_datasets()
    if args.src_mono_file:
        mono_data = get_mono_from_file(args.src_file)
        mono_data = mono_data[:args.num_sentence] if args.num_sentence>0 else mono_data
        tokenized_datasets = get_tokenized_datasets_mono(tokenizer, mono_data, args.src_lang_code, args.max_sentence_length, args.batch_size)
        tokenized_datasets = tokenized_datasets["train"]
    elif args.tokenized_datasets != "":    
        tokenized_datasets = torch.load(args.tokenized_datasets)[split]
        logger.info(tokenized_datasets)
    else:
        trans_para = get_translate_paras_from_file(args.src_file, args.tgt_file)
        trans_para = trans_para[:num_sentence+1] if num_sentence>0 else trans_para
        tokenized_datasets = get_tokenized_datasets(tokenizer,
                                                    trans_para,
                                                    args.src_lang_code, args.tgt_lang_code,
                                                    max_input_length=args.max_sentence_length,
                                                    max_target_length=args.max_sentence_length)
        tokenized_datasets = tokenized_datasets['train']
        
    dataloader = DataLoader(tokenized_datasets, batch_size=args.batch_size, collate_fn=data_collator)
    return dataloader

def init_exp(args):
    log_name = f"gene_trans_{args.num_sentence}.log" if not args.src_mono_file else f"gene_mono_trans_{args.num_sentence}.log"
    if not (type(args.num_sentence)==int):
        args.num_sentence = int(args.num_sentence)
    
    file_path = args.tokenized_datasets if args.tokenized_datasets != ""  else args.src_file
    des =  f"generate translations from {file_path}"
    if not args.src_mono_file:
        des += "翻译文本有reference"
    logger = initialize_exp(args, log_name, des)
    return logger

def check_params(args):
    
    assert os.path.exists(args.data_dir), 'data dir not exist'
    if args.tokenized_datasets != "":
        args.tokenized_datasets = os.path.join(args.data_dir, args.tokenized_datasets)
    else:
        assert args.src_file != ""
        args.src_file = os.path.join(args.data_dir, args.src_file)
        if not args.src_mono_file:
            assert args.tgt_file != ""
            args.tgt_file = os.path.join(args.data_dir, args.tgt_file)
    # logger.info(f"src file=>{args.src_file} \n tgt=>{args.tgt_file}")
    
    if args.num_beams <= 0:
        args.num_beams == None
    if not hasattr(args, "max_generate_length"):
        args.max_generate_length = args.max_length
    s = f"{args.source_lang}-{args.target_lang}-{args.num_sentence}" if args.num_sentence > 0 else f"{args.source_lang}-{args.target_lang}-all"
    args.save_path = os.path.join(args.saved_dir, s)
    pass
def main():
    args = parse_args()
    global logger
    check_params(args)
    logger = init_exp(args)

    tokenizer = load_tokenizer(args)
    data_collator = get_data_collator(args, tokenizer)
    model = get_model(args)
    model.eval()
    model = model.cuda()

    dataloader = get_dataloader(args, data_collator, tokenizer, args.num_sentence)
    
    logger.info("generate begin ......")
    src, pred, ref = [], [], []
    for x in tqdm(dataloader):
        for k in x.keys():
            x[k] = x[k].cuda() 
        if isinstance(model, M2M100ForConditionalGeneration):
            pred_outouts = model.generate(x['input_ids'], attention_mask = x['attention_mask'],
                                      max_length=args.max_length,forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang_code])
        else:
            pred_outouts = model.generate(x['input_ids'], attention_mask = x['attention_mask'],
                                      max_length=args.max_length)
            
        x["input_ids"][x["input_ids"]==-100] = tokenizer.pad_token_id
        src += tokenizer.batch_decode(x["input_ids"], skip_special_tokens=True)
        
        pred_outouts[pred_outouts==-100] = tokenizer.pad_token_id
        pred += tokenizer.batch_decode(pred_outouts, skip_special_tokens=True)
        if not args.src_mono_file:                                      #如果不是单语语料的翻译
            x['labels'][x['labels']==-100] = tokenizer.pad_token_id
            ref += tokenizer.batch_decode(x["labels"], skip_special_tokens=True)
        
        if args.num_sentence>0 and len(src) >= args.num_sentence:
            src = src[:args.num_sentence]
            pred = pred[:args.num_sentence]
            if not args.src_mono_file:
                ref = ref[:args.num_sentence]
            break
    logger.info(f"翻译完成，总共{len(src)}条句子")
    if args.src_mono_file:
        src_ref_pre = [SrcRefPreCor(s, pre=p) for s, p in zip(src, pred)]
    else:
        src_ref_pre = [SrcRefPreCor(s, r, p) for s, r, p in zip(src, ref, pred)]
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    torch.save(src_ref_pre, os.path.join(args.save_path, 'src_ref_pre.bin'))

    logger.info(f"三元组保存为{os.path.join(args.save_path, 'src_ref_pre.bin')}")
    logger.info("generate over")
    
main()
