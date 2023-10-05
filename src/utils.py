import argparse
from collections import Counter
from datetime import timedelta
import json
import math
import random
import time
from functools import partial
from typing import List
from datasets import Dataset, DatasetDict, load_dataset
from matplotlib import pyplot as plt
import numpy as np
import torch
import logging
import os
import evaluate
from tqdm import tqdm

from transformers import (Seq2SeqTrainingArguments,
                          AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          DataCollatorWithPadding,
                          MBartForConditionalGeneration,
                          M2M100ForConditionalGeneration,
                          AutoModelForSeq2SeqLM,
                          MT5ForConditionalGeneration
                          )

logger = logging.getLogger(__name__)


PRETRAINED_MODEL = (
                    # "microsoft/xlm-align-base",
                    # "facebook/mbart-large-cc25",
                    "facebook/mbart-large-50", 
                    "facebook/mbart-large-50-many-to-many-mmt",
                    "facebook/nllb-200-distilled-600M",
                    "mt5-small")

EVALUATE_METRICS = ("bleu",                                 # 使用scarebleu, tokenizer使用的是flores101的
                    "chrf",                                 # 使用chrf++
                    "ter",                                  # 使用sacrebleu库里的，大小写敏感的
                    "bleurt",                               # 使用推荐的checkpoint-20
                    )                               

FLORES_PATH = "/data/hyxu/codes/flores"              #评估数据集所在位置
OPUS_PATH = "/data/hyxu/codes/opus100"              #评估数据集所在位置
CACHE_DIR = "/data/hyxu/codes/cache_dir"

class SrcRefPreCor(object):
    """用来保存成对的src pre ref cor 的内容"""
    def __init__(self, src=None, ref=None, pre=None, cor=None) -> None:
        self.src = src if src else None
        self.ref = ref if ref else None
        self.pre = pre if pre else None
        self.cor = cor if cor else None
        pass
    def add_ref(self, ref):
        assert self.ref, "ref 不为空"
        self.ref = ref
    def add_pre(self, pre):
        assert self.pre, "pre 不为空"
        self.pre = pre
    def add_cor(self, cor):
        assert self.cor, "cor 不为空"
        self.cor = cor
    
    def __getitem__(self, i):
        if i==0:
            return self.src
        elif i==1:
            assert self.ref, f"i={i}, 取ref，但是ref为空"
            return self.ref
        elif i==2:
            assert self.pre, f"i={i}, 取pre，但是pre为空"
            return self.pre
        elif i==3:
            assert self.cor, f"i={i}, 取cor，但是cor为空"
            return self.cor
        else:
            assert -1<i<4, f"i的取值{i}, 无效"
    def __str__(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)
    def __repr__(self) -> str:
        return self.__str__()

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_lang", type=str, default="", help=""
    )
    parser.add_argument(
        "--src_lang_code", type=str, default="", help="tokenizer 中设置的语言代码"
    )
    parser.add_argument(
        "--target_lang", type=str
    )
    parser.add_argument(
        "--tgt_lang_code", type=str, default=""
    )
    parser.add_argument(
        "--model_name", type=str, default=PRETRAINED_MODEL[0], help="使用的预训练模型, 默认0是mbart50-large"
    )
    parser.add_argument(
        "--saved_dir", type=str, default="", help="数据或者模型 或者评估结果 保存的位置"
    )
    parser.add_argument(
        "--seed", type=int, default=10
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default="", help="如果有那个模型训练中断了的话，用此参数来重新加载checkpoints继续训练"
    )
    parser.add_argument(
        "--checkpoint", type=str, default='', help="这是要再训的 或者 评估的 模型的checkpoint"
    )
    parser.add_argument(
        "--optimer", type=str, default="", help="优化器，暂时不可设置"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="训练baseline用的是2e-5, retrain用的是4e-6"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="默认是16,"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=40, help="默认是40,"
    )
    parser.add_argument(
        "--max_sentence_length", type=int, default=256, help="训练中的句子最大长度"
    )
    parser.add_argument(
        "--max_generate_length", type=int, default=256, help="生成中的最大长度"
    )
    parser.add_argument(
        "--num_beams", type=int, default=0, help="暂时不改动"
    )
    parser.add_argument(
        "--data_dir", type=str, default="task/deep-encoder-shallow-decoder/data/en-ur", help="数据存放的位置"
    )
    parser.add_argument(
        "--tokenized_datasets", type=str, default="", help="如果为空，下面的file才起作用"
    )
    parser.add_argument(
        "--src_file", type=str, default="", help="必须包括valid的, 用','分隔"
    )
    parser.add_argument(
        "--tgt_file", type=str, default=""
    )
    parser.add_argument(
        "--tokenizer", type=str, default=""
    )
    parser.add_argument(
        "--data_collator", type=str, default=""
    )
    parser.add_argument(
        "--eval_steps", type=int, default=2000
    )
    parser.add_argument(
        "--save_steps", type=int, default=2000
    )
    parser.add_argument(
        "--evaluation_strategy", type=str, default="no"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=250
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2, help="与batch_size组合使用,一般默认为2"
    )
    parser.add_argument(
        "--label_smoothing_factor", type=float, default=0
    )
    parser.add_argument(
        "--pretrained_model", type=str, default="/public/home/hongy/pre-train_model", help="预训练模型的位置" 
    )
    parser.add_argument(
        "--bleu_tokenize", type=str, default="", help="目前用的sacrebleu，不可设"
    )
    parser.add_argument(
        "--evaluate_metrics", type=str, default="bleu", help="目前用的sacrebleu，不可设"
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=15
    )
    parser.add_argument(
        "--test_dataset", type=str, default="flores", help="在train和retrain里决定是否用flores的dev和test集"
    )
    parser.add_argument(
        "--name", type=str, default="cor10w", help="标识此次训练是干嘛的"
    )
    parser.add_argument(
        "--num_train_sentence", type=str, default="", help="再训练的使用的数据量，可以表示实验"
    )
    parser.add_argument(
        "--des", type=str, default="", help="写进log里描述再干什么"
    )
    parser.add_argument(
        "--freeze_decoder", type=lambda x: x=="true", default=False, help="在重新训练的时候是否要冻住decoder"
    )
    parser.add_argument(
        "--freeze_encoder", type=lambda x: x=="true", default=False,
        help="在重新训练的时候是否要冻住encoder, 主要用在训练en->方向"
    )
    args = parser.parse_args()
    return args


def avg(x):
    return sum(x) / len(x)

def try_gpu(i=0):
    if i <= torch.cuda.device_count():
        return torch.device(f"cuda:{i}")
    return torch.device('cpu')

def to_cuda(**params):

    return [p.cuda() for p in params]

def setup_seed(seed):
    torch.manual_seed(seed)                                 #不是返回一个生成器吗？
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True               #使用确定性的卷积算法

def create_logger(filepath=None, rank=0, name=None):
    """
    Create a logger.
    Use a different log file for each process.
    filepath 为None的时候即不输出到文本里面去，
    rank为0的时候即单线程
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    if name != None:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def get_datasets_from_flores(src_lang_code, tgt_lang_code):
    """给src_lang_code 和tgt_lang_code 返回flores的valid 和test 数据集"""
    # tokenized_datasets = load_dataset(FLORES_PATH, f"{src_lang_code}-{tgt_lang_code}", cache_dir=CACHE_DIR)
    tokenized_datasets = load_dataset(FLORES_PATH, f"{src_lang_code}-{tgt_lang_code}")
    tokenized_datasets['test'] = tokenized_datasets.pop('devtest')
    tokenized_datasets['valid'] = tokenized_datasets.pop('dev')
    return tokenized_datasets
def get_datasets_from_opus(src_file, tgt_file):
    "src_file和tgt_file 是从数据集里加载句子的依据"
    tag = f"{src_file}-{tgt_file}" if src_file<tgt_file else f"{tgt_file}-{src_file}"
    tokenized_datasets = load_dataset(OPUS_PATH, tag, cache_dir=CACHE_DIR)
    tokenized_datasets.pop("train")
    tokenized_datasets['valid'] = tokenized_datasets.pop("validation")
    for split in ("valid", "test"):
        src, tgt = [], []
        for i in tokenized_datasets[split]['translation']:
            src.append(i[src_file])
            tgt.append(i[tgt_file])
        data = Dataset.from_dict({src_file:src, tgt_file:tgt})
        tokenized_datasets[split] = data
    return tokenized_datasets

def preprocess_function(examples, src_lang, tgt_lang, tokenizer, max_input_length, max_target_length):
    inputs = [ex for ex in examples[src_lang]]
    targets = [ex for ex in examples[tgt_lang]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Set up the tokenizer for targets 源语言与目标语言使用联合词典的
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    # model_inputs["labels_attention_mask"] = labels["attention_mask"]
    return model_inputs

def get_tokenized_datasets(tokenizer, trans_para, src_lang, tgt_lang, max_input_length, max_target_length, batch_size=None):
    """
    注意 着里的trans_para 只能是有两个元素的，分别作为源语言和目标语言, 也可以是datasetdict
    只进行tokenized不做split trans_para 可以是list也可以是DatasetDict
    """
    batch_tokenize_fn = partial(preprocess_function,
                                tokenizer=tokenizer,
                                src_lang=src_lang,
                                tgt_lang=tgt_lang,
                                max_input_length=max_input_length,
                                max_target_length=max_target_length,
                                )
    if not isinstance(trans_para, DatasetDict):
        trans_para = {
            src_lang: [src for src, _ in trans_para],
            tgt_lang: [tgt for _, tgt in trans_para]
        }
        raw_datasets = Dataset.from_dict(trans_para)
        raw_datasets = DatasetDict({'train': raw_datasets})
    else:
        raw_datasets = trans_para
    remove_names = raw_datasets['train'].column_names if "train" in raw_datasets else raw_datasets['test'].column_names

    tokenized_datasets = raw_datasets.map(batch_tokenize_fn, batched=True, batch_size=batch_size,
                                          remove_columns=remove_names)
    return tokenized_datasets

def get_src_ref_pre_cor_paras_from_file(*files):
    """_summary_
        files 的顺序必须是src， ref， pre， cor，后面的可以为空，但前面的必需有
    Returns:
        返回[[src],[ref], ...]
    """
    file_data = []
    for i, path in enumerate(files):
        with open(path, 'r') as f:
            f_data = f.readlines()
        f_data = [s.rstrip('\n').rstrip(" ") for s in f_data]
        file_data.append(f_data)

    # 过滤掉句子长度为0的句子
    trans_para = [item for item in zip(*file_data) if all([len(x)>0 for x in item])]
    # 将元组转换成对象 考虑效率的问题，以后再说是否转换成对象
    trans_para = [SrcRefPreCor(*item) for item in trans_para]
    return trans_para

def get_translate_paras_from_file(src_file, tgt_file):
    trans_paras = get_src_ref_pre_cor_paras_from_file(src_file, tgt_file)   #这里tgt_file作为ref传进去的
    return [[p[0], p[1]] for p in trans_paras]

def get_mono_from_file(src_f):
    mono_data = get_src_ref_pre_cor_paras_from_file(src_f)
    return [p[0] for p in mono_data]

## TODO get_tokenized_data_mono  可以使用DataCollatorWithPadding
def get_tokenized_datasets_mono(tokenizer, mono, src_lang, max_input_length, batch_size):
    if not isinstance(mono, DatasetDict):
        mono = {
            src_lang: [src for src in mono],
            }
        raw_datasets = Dataset.from_dict(mono)
        raw_datasets = DatasetDict({'train': raw_datasets})
    else:
        raw_datasets = mono
    remove_names = raw_datasets['train'].column_names if "train" in raw_datasets else raw_datasets['test'].column_names
    def mono_preprocess_function(examples, src_lang, tokenizer, max_sentence_length):
        inputs = [ex for ex in examples[src_lang]]
        model_inputs = tokenizer(inputs, max_length=max_sentence_length, truncation=True)
        return model_inputs
    batch_tokenize_fn = partial(mono_preprocess_function,
                            tokenizer=tokenizer,
                            src_lang=src_lang,
                            max_sentence_length=max_input_length,
                            )
    tokenized_datasets = raw_datasets.map(batch_tokenize_fn, batched=True, batch_size=batch_size,
                                          remove_columns=remove_names)
    return tokenized_datasets

def split_datasets(dataset, test=3000, valid=0, seed=10):
    """如果valid是0 那么就之分train 和 test 不分 valid"""
    if isinstance(dataset, Dataset):
        split_dataset_dict = dataset.train_test_split(test_size=test, seed=seed)
    elif isinstance(dataset, DatasetDict):
        split_dataset_dict = dataset['train'].train_test_split(test_size=test, seed=seed)
    if valid != 0:
        valid_dataset = split_dataset_dict.pop("test")
        split_dataset_dict = split_dataset_dict['train'].train_test_split(test_size=valid, seed=seed)
        split_dataset_dict['valid'] = valid_dataset
    return split_dataset_dict

def paras_filter_by_belu(correct_paras, bleu=5, patience=-1, high=101, return_bleu=False):
    """correct_paras里的内容会原封不动的返回去 大于patience小于high 不包含等于"""
    data = [(pa, b) for pa, b in zip(correct_paras, bleu) if b>patience and b<high]
    if return_bleu:
        return [p for p, b in data], [b for p, b in data]
    else:
        return [p for p, b in data]

def trans_filter_by_len(paras, r_p_bleu=None, l_len=7, per_len=0.4, return_bleu=False):
    """len 过滤条件，
        对于src_ref_pre 来说 ref和pre 词的个数大于7且句子的词数差别小于30%
        对于src_pre 来说 src和pre 词的个数大于8且句子的词数差别小于40% 这就要换一个函数了
    """
    assert r_p_bleu == None and return_bleu == False
    def fn(src, tgt):
        l1 = len(src.split(" "))
        l2 = len(tgt.split(" "))
        if l1 > l_len and l2 > l_len:
            if (abs(l1-l2)/l1) <= per_len:
                return True
        return False
    if r_p_bleu != None and return_bleu:
        data = [(paras[i], b) for i, b in enumerate(r_p_bleu) if fn(paras[i].ref, paras[i].pre)]
    else:
        data = [p for p in paras if fn(p.src, p.pre)]
    if return_bleu:
        return data
    else:
        return data


def load_tokenizer(args):
    """当args中需要有args.src_lang_code与args.tgt_lang_code"""
    assert hasattr(args, "src_lang_code") and hasattr(args, "tgt_lang_code")
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint != '':
        path = os.path.join(args.resume_from_checkpoint)
    elif hasattr(args, 'checkpoint') and args.checkpoint != "":
        path = args.checkpoint
    else:
        logger.info(args.model_name)
        logger.info(PRETRAINED_MODEL)
        assert args.model_name in PRETRAINED_MODEL, "model don't load"
        path = os.path.join(args.pretrained_model, args.model_name.split('/')[-1])
    logger.critical(path)

    # tokenizer = AutoTokenizer.from_pretrained(path, src_lang=args.src_lang_code, tgt_lang=args.tgt_lang_code)
    tokenizer = AutoTokenizer.from_pretrained(path)

    tokenizer.src_lang = args.src_lang_code
    tokenizer.tgt_lang = args.tgt_lang_code
    logger.info(f"load tokenizer form {path}")
    logger.info(tokenizer)
    return tokenizer

def initialize_exp(args, log_name='train.log', des='这是训练nmt基线的实验'):
    if not hasattr(args, 'saved_dir'):
        assert hasattr(args, 'output_dir')
        args.saved_dir = args.output_dir
    if not hasattr(args, "resume_from_checkpoint") or args.resume_from_checkpoint == "":
        if not os.path.exists(args.saved_dir):
            os.mkdir(args.saved_dir)
    with open(os.path.join(args.saved_dir, log_name), 'w') as f:
        f.write("")
    logger = create_logger(os.path.join(args.saved_dir, log_name), rank=getattr(args, 'global_rank', 0))
    logger.info(f"============ Initialized logger 当前进程号为: {os.getpid()} ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(args)).items())))
    logger.info("The experiment will be stored in %s\n" % args.saved_dir)
    logger.info("The log file name is %s\n" % log_name)
    logger.critical(des)
    return logger

def save_flores_test_as_file(save_path, src_lang_code, tgt_lang_code, mode="txt"):
    """将flores-200里的test数据保存为文件,默认保存为txt模型，可更改为List[SrcRefPre]的bin文件

    Args:
        save_path (_type_): _description_
        src_lang_code (_type_): _description_
        tgt_lang_code (_type_): _description_
    """
    data = load_dataset(FLORES_PATH, f"{src_lang_code}-{tgt_lang_code}", cache_dir=CACHE_DIR)
    logger.info(data)
    data = data['devtest'].to_dict()
    src_data, tgt_data = data[f"sentence_{src_lang_code}"], data[f"sentence_{tgt_lang_code}"]
    assert len(src_data) == len(tgt_data)
    
    if mode != "txt":
        data = [SrcRefPreCor(src, ref) for src, ref in zip(src_data, tgt_data)]
        torch.save(data, os.path.join(save_path, f"flores.{src_lang_code[:3]}-{tgt_lang_code[:3]}-test.bin"))
    else:
        src_data = [s+"\n" for s in src_data]
        tgt_data = [s+"\n" for s in tgt_data]
        with open(os.path.join(save_path, f"flores.{src_lang_code[:3]}-{tgt_lang_code[:3]}-test.{src_lang_code[:3]}"), 'w') as f:
            f.writelines(src_data)
        with open(os.path.join(save_path, f"flores.{src_lang_code[:3]}-{tgt_lang_code[:3]}-test.{tgt_lang_code[:3]}"), 'w') as f:
            f.writelines(tgt_data)
    logger.info("保存完成")

def get_data_collator(args, tokenizer, model=None):
    """可以在这里自定datacollator"""
    if hasattr(args, "data_collator") and args.data_collator != "":
        return torch.load(args.data_collator)
    if hasattr(args, "src_mono_file") and args.src_mono_file :
        data_collator = DataCollatorWithPadding(tokenizer, padding=True, max_length=args.max_sentence_length)
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, model=model, max_length=args.max_sentence_length)
    return data_collator

def get_model(args, config=None, forced_bos_token_id=None):
    model_type = AutoModelForSeq2SeqLM
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint != "":
        path = args.resume_from_checkpoint
        model_type = AutoModelForSeq2SeqLM
    elif hasattr(args, 'checkpoint') and args.checkpoint != "":
        path = args.checkpoint
        model_type = AutoModelForSeq2SeqLM
    else:
        path = os.path.join(args.pretrained_model, args.model_name.split('/')[-1])
        if args.model_name == PRETRAINED_MODEL[0]:
            model_type = MBartForConditionalGeneration
        elif args.model_name == PRETRAINED_MODEL[2]:
            model_type = M2M100ForConditionalGeneration
        elif args.model_name == PRETRAINED_MODEL[3]:
            model_type = MT5ForConditionalGeneration
    logger.info(path)
    
    model = model_type.from_pretrained(path)
    
    if hasattr(args, "freeze_decoder") and args.freeze_decoder:
        decoder = model.get_decoder()
        for name, param in decoder.named_parameters():
            param.requires_grad = False
    if hasattr(args, "freeze_encoder") and args.freeze_encoder:
        encoder = model.get_encoder()
        for name, param in encoder.named_parameters():
            param.requires_grad = False

            
    logger.critical("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))
    logger.info(model)
    return model

def get_training_args(args):
    
    logger.info(f"实验中数据存在此中：{args.saved_dir}")
    if hasattr(args, "resume_from_checkpoint") and args.resume_from_checkpoint != "":
        training_args = torch.load(os.path.join(args.resume_from_checkpoint, "training_args.bin"))
    # elif hasattr(args, "checkpoint") and args.checkpoint != "":
    #     training_args = torch.load(os.path.join(args.checkpoint, 'training_args.bin'))
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.saved_dir,
            evaluation_strategy=args.evaluation_strategy if hasattr(args, "evaluation_strategy") else  "steps",
            learning_rate=args.lr if hasattr(args, "lr") else 2e-5,
            per_device_eval_batch_size=args.eval_batch_size if hasattr(args, "eval_batch_size") and args.eval_batch_size>args.batch_size else args.batch_size,
            per_device_train_batch_size=args.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=50,
            generation_max_length=args.max_generate_length if hasattr(args, "max_generate_length") else 256,
            generation_num_beams=args.num_beams if hasattr(args, "num_beams") and args.num_beams and args.num_beams>0 else None,
            seed=args.seed,
            predict_with_generate=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps if hasattr(args, "gradient_accumulation_steps") else 1,
            fp16=args.fp16 if hasattr(args, "fp16") else True,
            fp16_opt_level=args.fp16_opt_level if hasattr(args, "fp16_opt_level") else "O3",
            half_precision_backend="auto",
            label_smoothing_factor=args.label_smoothing_factor if hasattr(args, "label_smoothing_factor") else 0,
            load_best_model_at_end=True,
            eval_steps=args.eval_steps if hasattr(args, "eval_steps") else 5000,
            save_steps=args.save_steps if hasattr(args, "save_steps") else 5000,
            warmup_steps=args.warmup_steps if hasattr(args, "warmup_steps") else 100,
            logging_steps=args.logging_steps if hasattr(args, "logging_steps") else 500,
            dataloader_num_workers=args.dataloader_num_workers if hasattr(args, "dataloader_num_workers") else 0,
            metric_for_best_model="bleu",
            report_to=['tensorboard'],
        )
    logger.info(training_args)
    return training_args

def compute_metric(ref, pre, metric=None):
    if metric==None:
        metric = evaluate.load('sacrebleu')
    # bleu = metric.compute(predictions=[pre], references=[[ref]], tokenize="flores200")
    bleu = metric.compute(predictions=[pre], references=[[ref]])
    return bleu

def compute_batch(ref, pre, metric_name="bleu"):
    """暂时只能默认的使用metric_name = bleu的"""
    if metric_name == EVALUATE_METRICS[0]:
        metric = evaluate.load('sacrebleu')
    elif metric_name == EVALUATE_METRICS[2]:
        metric = evaluate.load("ter")
    else:
        assert metric_name == EVALUATE_METRICS[0]
    bleu = []
    for r, p in tqdm(list(zip(ref, pre))):
        bleu.append(compute_metric(r, p, metric)['score'])
    return bleu

def compute_chrf(predictions, references, word_order=2):
    logger.info("\n")
    logger.info(f"==== 开始计算chrf++ ====")
    metric = evaluate.load('chrf')
    res = metric.compute(predictions=predictions, references=references, word_order=word_order)
    logger.critical(f"results is {res}")
    return res
def compute_bleu(predictions, references, tokenize=""):
    logger.info("\n")
    logger.info(f"==== 开始计算sacrebleu ====")
    metric = evaluate.load('sacrebleu')
    if tokenize=="":
        res = metric.compute(predictions=predictions, references=references)
    else:
        res = metric.compute(predictions=predictions, references=references, tokenize=tokenize)
    logger.critical(f"results is {res}")
    return res
def compute_ter(predictions, references):
    logger.info("\n")
    logger.info(f"==== 开始计算ter ====")
    metric = evaluate.load('ter')
    res = metric.compute(predictions=predictions, references=references, case_sensitive=True)
    logger.critical(f"results is {res}")
    return res
def compute_bleurt(predictions, references, batch_size=128):
    ## TODO 有待补充
    logger.info("==== 开始计算bleurt ====")
    # assert isinstance(ref[0], list), "目前bleurt只支持一个ref的计算"
    from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
    
    references = [r[0] for r in references]
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20')
    # tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

    result = []
    num_steps = len(predictions) // batch_size if len(predictions) % batch_size == 0 else len(predictions) // batch_size + 1
    model.eval()
    model = model.cuda()
    with torch.no_grad():
        for step in tqdm(range(num_steps)):
            pre = predictions[step*batch_size:(step+1)*batch_size]
            ref = references[step*batch_size:(step+1)*batch_size]
            
            inputs = tokenizer(ref, pre, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = inputs.to(torch.device("cuda:0"))
            res = model(**inputs).logits.flatten().tolist()
        
            result += res
    logger.info(f"results is {avg(result)}")
    return {"score": avg(result)}
    
def get_compute_metrics(args, tokenizer):
    args.bleu_tokenize = "flores200" if args.bleu_tokenize == "" and args.test_dataset =="flores" else args.bleu_tokenize
    args.evaluate_metrics = [m for m in args.evaluate_metrics if m in EVALUATE_METRICS]
    compute_fn_dict = dict()
    if EVALUATE_METRICS[0] in args.evaluate_metrics:                # bleu
        compute_fn_dict['bleu'] = partial(compute_bleu,tokenize=args.bleu_tokenize)
    if EVALUATE_METRICS[1] in args.evaluate_metrics:
        compute_fn_dict['chrf'] = compute_chrf             # chrf evaluate.load("chrf")
    if EVALUATE_METRICS[2] in args.evaluate_metrics:
        compute_fn_dict['ter'] = compute_ter               # chrf evaluate.load("chrf")
    if EVALUATE_METRICS[3] in args.evaluate_metrics:
        compute_fn_dict['bleurt'] = compute_bleurt         # lucadiliello/BLEURT-20
        # assert 1==2, "此指标暂时未加入使用"
    
    def compute_metrics(eval_preds):

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_lables = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_lablels = [[label.strip()] for label in decoded_lables]
        result = {k: v(predictions=decoded_preds, references=decoded_lablels)["score"] \
                  for k, v in compute_fn_dict.items()}
        # result = metric.compute(predictions=decoded_preds, references=decoded_lablels)
        # return {'bleu': result['score']}
        return result
    return compute_metrics

def get_len_dis(src_len, tgt_len):
    """统计src和tgt文件对应句子的长度差的绝对值的平均和分布"""
    
    len_dis = [abs(s-t) for s, t in zip(src_len, tgt_len)]
    counter = Counter(len_dis)
    item = sorted(counter.items(), key=lambda x: x[0])
    plt.plot([i[0] for i in item], [math.log10(i[1]) for i in item])
    plt.legend(["num of dis between src and tgt"])
    plt.show()
    
    len_dis_precent = [int((abs(s-t)/s)*100)  for s, t in zip(src_len, tgt_len)]        # 所差的个数占src句子的比例
    len_dis_precent = [p for p in len_dis_precent if p<500]                            # 去掉异常值
    counter_pre = Counter(len_dis_precent)
    item = sorted(counter_pre.items(), key=lambda x: x[0])
    plt.plot([i[0] for i in item], [math.log10(i[1]) for i in item])
    plt.legend(["percent of dis num in src and tgt"])
    plt.show()
    return sum(len_dis) / len(len_dis), (counter, counter_pre)

def get_sentence_len_from_file(src_p, tgt_p):
    with open(src_p, 'r') as src_f, open(tgt_p, "r") as tgt_f:
        src_data = src_f.readlines()
        tgt_data = tgt_f.readlines()
    src_len = map(lambda x: len(x.split(" ")), src_data)
    tgt_len = map(lambda x: len(x.split(" ")), tgt_data)
    return list(src_len), list(tgt_len)

def plot_len_fre_form_file(*path):
    """画出文件中句子的词的个数"""
    file_datas = []
    for p in path:
        assert os.path.exists(p)
        with open(p, 'r') as f:
            data = f.readlines()
        file_datas.append(data)
    def fn(s):
        l1 = len(s.split(" "))
        return l1 if l1 < 200 else 200
    sentence_len = [[fn(s) for s in data] for data in file_datas]
    len_counter = [Counter(s_len) for s_len in sentence_len]
    len_fre_item = [sorted(counter.items(), key=lambda x: x[0]) for counter in len_counter]       # 0是长度， 1是频次
    for item in len_fre_item:
        plt.plot([x[0] for x in item], [math.log10(x[1]) for x in item])
    plt.legend(path)
    plt.show()

"""下面的三个加噪音函数copy from => fairseq.data.denoising_dataset.py DenoisingDataset"""
def add_permuted_noise(tokens, p):
    num_words = len(tokens)
    num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
    substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
    tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
    return tokens

def add_rolling_noise(tokens):
    offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
    tokens = torch.cat(
        (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
        dim=0,
    )
    return tokens

# def add_insertion_noise(self, tokens, p):
#     if p == 0.0:
#         return tokens

#     num_tokens = len(tokens)
#     n = int(math.ceil(num_tokens * p))

#     noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
#     noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
#     noise_mask[noise_indices] = 1
#     result = torch.LongTensor(n + len(tokens)).fill_(-1)

#     num_random = int(math.ceil(n * self.random_ratio))
#     result[noise_indices[num_random:]] = self.mask_idx
#     result[noise_indices[:num_random]] = torch.randint(low=1, high=len(self.vocab), size=(num_random,))

#     result[~noise_mask] = tokens

#     assert (result >= 0).all()
#     return result


