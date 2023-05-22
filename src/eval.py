"""
在给定测试集合上评估模型性能
目前支持的模型有[mbart-large-50]

"""
import numpy as np
import random
import torch
import os
import evaluate
import argparse
from transformers import (AutoTokenizer,
                          Seq2SeqTrainer,
                          DataCollatorForSeq2Seq,
                          M2M100ForConditionalGeneration
                          )

from utils import (PRETRAINED_MODEL, create_logger, get_datasets_from_flores,
                   get_datasets_from_opus, parse_args, get_translate_paras_from_file,
                   get_tokenized_datasets, initialize_exp, load_tokenizer,
                   get_model, get_compute_metrics, get_training_args)
from logging import getLogger

logger = getLogger(__name__)

def load_tokenized_datasets(args, tokenizer, split='test'):
    if args.test_dataset == "flores":
        # from datasets import load_dataset
        # tokenized_datasets = load_dataset(args.data_dir, f"{args.src_file}-{args.tgt_file}")
        # tokenized_datasets['test'] = tokenized_datasets.pop('devtest')
        # tokenized_datasets['valid'] = tokenized_datasets.pop('dev')
        tokenized_datasets = get_datasets_from_flores(args.src_file, args.tgt_file)
        datasets = get_tokenized_datasets(tokenizer, trans_para=tokenized_datasets, 
                                            src_lang=f"sentence_{args.src_file}", tgt_lang=f"sentence_{args.tgt_file}",
                                            max_input_length=args.max_sentence_length, max_target_length=args.max_sentence_length)
        datasets = datasets[split]
    elif args.test_dataset == "opus":
        tokenized_datasets = get_datasets_from_opus(args.src_file, args.tgt_file)
        datasets = get_tokenized_datasets(tokenizer, trans_para=tokenized_datasets, 
                                            src_lang=args.src_file, tgt_lang=args.tgt_file,
                                            max_input_length=args.max_sentence_length, max_target_length=args.max_sentence_length)
        datasets = datasets[split]
    else:
        trans_para = get_translate_paras_from_file(args.src_file, args.tgt_file)
        datasets = get_tokenized_datasets(tokenizer,trans_para,
                                          args.src_lang_code, args.tgt_lang_code,
                                          args.max_sentence_length, args.max_sentence_length )
        datasets = datasets['train']
    logger.info(datasets)
    return datasets

def get_data_collator(args, tokenizer, model=None):
    """可以在这里自定datacollator"""
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, model=model,
                                        max_length=args.max_sentence_length)

    return data_collator

def get_eval_args(args):
    eval_args = get_training_args(args)
    return eval_args

def init_exp(args):
    file = f"{args.test_dataset}-src:{args.src_file}\n tgt:{args.tgt_file}"
    des = f'这个实验的评估文件是{file}'
    log_name = f"{args.test_dataset}-"
    log_name += "-".join(args.evaluate_metrics)
    logger = initialize_exp(args, log_name, des)
    return logger

def check_params(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert os.path.exists(args.checkpoint), '此checkpoint 不存在'
    if args.saved_dir == "":
        args.saved_dir = args.checkpoint
    else:
        if not os.path.exists(args.saved_dir):
            os.mkdir(args.saved_dir)
    if args.test_dataset in ("flores", "opus"):
        # assert args.source_lang in args.src_file and args.target_lang in args.tgt_file
        assert "/" not in args.src_file and "/" not in args.tgt_file
    else:
        assert os.path.isdir(args.data_dir), '文件夹错位'
        assert args.src_file != "" and args.tgt_file != ""
        args.src_file = os.path.join(args.data_dir, args.src_file)
        args.tgt_file = os.path.join(args.data_dir, args.tgt_file)
    
    if not hasattr(args, 'max_generate_length'):
        args.max_generate_length = args.max_length
        
    if args.num_beams <= 0:             # 设置为默认的，如果此时给training args的是None的话他就会去找model config中的
        args.num_beams = None    
    args.bleu_tokenize = "flores200" if args.bleu_tokenize and args.test_dataset=="flores" else args.bleu_tokenize
    args.evaluate_metrics = [m for m in args.evaluate_metrics.split(",") if len(m)>1]
    pass

def main():
    args = parse_args()
    check_params(args)
    global logger
    logger = init_exp(args)
    tokenizer = load_tokenizer(args)
    tokenized_datasets = load_tokenized_datasets(args, tokenizer)

    eval_args = get_eval_args(args)
    
    compute_metrics = get_compute_metrics(args, tokenizer)
    model = get_model(args)
    data_collator = get_data_collator(args, tokenizer, model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    logger.info('evaluating')
    if isinstance(model, M2M100ForConditionalGeneration):
        logger.info(f"M2M100ForConditionalGeneration")
        test_output = trainer.predict(tokenized_datasets,
                                   forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang_code])
        # test_ouptut = trainer.evaluate(tokenized_datasets, metric_key_prefix="test")
    else:
        test_output = trainer.predict(tokenized_datasets)
    def _decode(x):
        x[x==-100] = tokenizer.pad_token_id
        return tokenizer.batch_decode(x, skip_special_tokens=True)
    predictions = [p+"\n" for p in _decode(test_output.predictions)]
    label_ids = [p+"\n" for p in _decode(test_output.label_ids)]
    
    with open(os.path.join(args.saved_dir, "predictions.txt"), "w") as  f:
        f.writelines(predictions)
    with open(os.path.join(args.saved_dir, "references.txt"), "w") as  f:
        f.writelines(label_ids)

    logger.info(f"test_output => {test_output.metrics}")

main()