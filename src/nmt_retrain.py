"""
训练一个NMT模型， 现在支持的预训练模型有:
[facebook/mbart-large-cc25, microsoft/xlm-align-base,facebook/mbart-large-50, facebook/mbart-large-50-many-to-many-mmt]
"""

import torch
import os
import argparse

from transformers import (AutoConfig,
                          EarlyStoppingCallback,
                          Seq2SeqTrainer,
                          M2M100ForConditionalGeneration,
                          )

from datasets import Dataset, DatasetDict

from utils import (create_logger, 
                   PRETRAINED_MODEL, get_datasets_from_flores, get_datasets_from_opus,
                   load_tokenizer,
                   get_translate_paras_from_file,
                   get_tokenized_datasets,
                   get_data_collator,
                   get_compute_metrics,
                   get_model,
                   get_training_args,
                   initialize_exp,
                   parse_args
                   )

logger = create_logger(name=__name__)

os.environ["WANDB_DISABLED"] = "true"

def check_params(args):
    def exists(f):
        assert os.path.exists(f), f"路径：{f} 不存在"

    if args.tokenized_datasets != '':
        args.tokenized_datasets = os.path.join(args.data_dir, args.tokenized_datasets)
        exists(args.tokenized_datasets)
    else:
        assert args.src_file != "" and args.tgt_file != ""
        args.src_file = [s for s in args.src_file.split(",") if len(s)>0]
        args.tgt_file = [s for s in args.tgt_file.split(",") if len(s)>0]
        if args.test_dataset in ("opus", "flores"):
            args.src_file[0] = os.path.join(args.data_dir, args.src_file[0])
            args.tgt_file[0] = os.path.join(args.data_dir, args.tgt_file[0])
        else:
            args.src_file = [os.path.join(args.data_dir, s) for s in args.src_file]
            args.tgt_file = [os.path.join(args.data_dir, s) for s in args.tgt_file]
    if args.checkpoint == "":
        args.tokenizer = os.path.join(args.data_dir, args.tokenizer)
        exists(args.tokenizer)

    if args.data_collator != "":
        args.data_collator = os.path.join(args.data_dir, args.data_collator)
        exists(args.data_collator)

    assert not (args.freeze_encoder and args.freeze_decoder),"只能同时冻住encoder或者deocer一个"
    
    args.bleu_tokenize = "flores200" if args.test_dataset=="flores" and args.bleu_tokenize == "" else args.bleu_tokenize
    
    last_fix = args.name 
    last_fix += "_encoder" if args.freeze_decoder else ""
    last_fix += "_decoder" if args.freeze_encoder else ""
    args.saved_dir = os.path.join(args.saved_dir,
                          f"{args.source_lang}-{args.target_lang}-chpk",f"retrain-{last_fix}")
    args.evaluate_metrics = args.evaluate_metrics.split(",")


def load_tokenized_datasets(args, tokenizer=None):

    if args.tokenized_datasets != "":
        path = args.tokenized_datasets
        logger.info(f"load tokenized_datasets form {path}")
        datasets = torch.load(path)
    elif len(args.src_file) > 1:
        def get_dataset(src_f, tgt_f, batch_size=None):
            trans_para = get_translate_paras_from_file(src_f, tgt_f)
            datasets = get_tokenized_datasets(tokenizer, trans_para, args.src_lang_code, args.tgt_lang_code,
                                              max_input_length=args.max_sentence_length,
                                              max_target_length=args.max_sentence_length,batch_size=batch_size)
            return datasets['train']
        splits = ['train', 'valid', 'test'] if len(args.src_file)==3 else ['train', 'valid']
        datasets = DatasetDict()
        if args.test_dataset == "flores":
            datasets = get_datasets_from_flores(args.src_file[1], args.tgt_file[1])
            datasets= get_tokenized_datasets(tokenizer, datasets, f"sentence_{args.src_file[1]}", f"sentence_{args.tgt_file[1]}",
                                              max_input_length=args.max_sentence_length,
                                              max_target_length=args.max_sentence_length,batch_size=args.eval_batch_size)
            datasets[splits[0]] = get_dataset(args.src_file[0], args.tgt_file[0])
        elif args.test_dataset == "opus":
            datasets = get_datasets_from_opus(args.src_file[1], args.tgt_file[1])
            datasets = get_tokenized_datasets(tokenizer, trans_para=datasets, 
                                            src_lang=args.src_file[1], tgt_lang=args.tgt_file[1],
                                            max_input_length=args.max_sentence_length,
                                            max_target_length=args.max_sentence_length,batch_size=args.eval_batch_size)
            datasets[splits[0]] = get_dataset(args.src_file[0], args.tgt_file[0], args.batch_size)
        else:
            for src_f, tgt_f, split in zip(args.src_file, args.tgt_file, splits):
                datasets[split] = get_dataset(src_f, tgt_f, args.batch_size)
    else:
        pass
    logger.info(datasets)
    return datasets

def set_model_config(args, tokenizer):
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint != "":
        path = args.resume_from_checkpoint
    elif hasattr(args, "checkpoint") and args.checkpoint != "":
        path = args.checkpoint
    else:
        assert args.model_name in PRETRAINED_MODEL
        path = os.path.join(args.pretrained_model, args.model_name.split('/')[-1])
    config = AutoConfig.from_pretrained(path)
    logger.info(f"load model config form {path}")
    logger.info(config)
    return config
def set_generation_config(args, tokenizer, model):
    model.generation_config.forced_bos_token_id = tokenizer.lang_code_to_id[args.tgt_lang_code]
    pass
def init_exp(args):
    des = f"从{args.checkpoint}进行再次训练" + args.des
    logger = initialize_exp(args, log_name='train.log', des=des)
    return logger

def main():

    args = parse_args()
    check_params(args)
    global logger
    
    logger = init_exp(args)

    tokenizer = load_tokenizer(args)
    tokenized_datasets = load_tokenized_datasets(args, tokenizer)
    config = set_model_config(args, tokenizer)

    training_args = get_training_args(args)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),]
    compute_metrics = get_compute_metrics(args, tokenizer)
    ## ! 加载模型
    model = get_model(args, config)
    data_collator = get_data_collator(args, tokenizer, model=model)
    
    if isinstance(model, M2M100ForConditionalGeneration):                              ## ! 似乎在这里指定有用
        set_generation_config(args, tokenizer, model)

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['valid'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    logger.critical('training')
    if not hasattr(args, "resume_from_checkpoint") or args.resume_from_checkpoint == "":
        trainer_output = trainer.train()
    else:
        trainer_output = trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

    logger.info(f"trainer_output => {trainer_output}")
    
    logger.critical("为args的metrics加上ter、chrf")
    args.evaluate_metrics += ["ter", "chrf", "bleurt"]
    trainer.compute_metrics = get_compute_metrics(args, tokenizer)
    
    logger.info('evaluating')
    
    if isinstance(model, M2M100ForConditionalGeneration): 
        eval_output = trainer.predict(tokenized_datasets["test"], forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang_code])
    else:
        eval_output = trainer.predict(tokenized_datasets["test"])

    logger.info(f"eval_output => {eval_output.metrics}")
    pass

main()