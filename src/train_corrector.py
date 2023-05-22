
from transformers import (Seq2SeqTrainer,
                          EarlyStoppingCallback,
                          )

from utils import(create_logger,
                  get_translate_paras_from_file,
                  get_tokenized_datasets,
                  split_datasets,
                  try_gpu,
                  to_cuda,
                  paras_filter_by_belu,
                  PRETRAINED_MODEL,
                  load_tokenizer,
                  initialize_exp,
                  get_data_collator,
                  get_training_args,
                  get_compute_metrics,
                  get_model,
                  )

from torch.utils.data import DataLoader
import os
import argparse
import evaluate

os.environ["WANDB_DISABLED"] = "true"

logger = create_logger(name=__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang", type=str, default='en'
    )
    parser.add_argument(
        "--lang_code", type=str, default="en_XX"
    )
    parser.add_argument(
        "--model_name", type=str, default=PRETRAINED_MODEL[0]
    )
    parser.add_argument(
        "--pretrained_model", type=str, default="./"
    )
    parser.add_argument(
        "--saved_dir", type=str, default="/checkpoint-corrector/"
    )
    parser.add_argument(
        "--seed", type=int, default=10
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=""
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )
    parser.add_argument(
        "--max_sentence_length", type=int, default=256
    )
    parser.add_argument(
        "--max_generate_length", type=int, default=256
    )
    parser.add_argument(
        "--data_dir", type=str, default=""
    )
    parser.add_argument(
        "--evaluate_metrics", type=str, default="bleu,"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=4000,
    )
    parser.add_argument(
        "--save_steps", type=int, default=4000,
    )
    parser.add_argument(
        "--metrics_patience", type=str, default="10,"
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=10
    )
    parser.add_argument(
        "--num_beams", type=int, default=1
    )
    args = parser.parse_args()
    return args


def compute_metrics(args):
    """返回每条句子的分数和平行语句"""
    correct_paras = get_translate_paras_from_file(args.references, args.hypothesis)
    logger.info(f"------------------开始计算bleu 共{len(correct_paras)} 条平行语句----------------")
    metric = evaluate.load('sacrebleu')
    bleu = []
    for step, pa in enumerate(correct_paras):
        re, py = pa
        result = metric.compute(references=[[re]], predictions=[py])
        bleu.append(result['score'])

        if step % 100 == 0:
            logger.info(f"第{step} bleu =》{result}")
    logger.info(f"------------------bleu 分数计算完成 平均分为 {sum(bleu) / len(bleu)}")
    return bleu, correct_paras

def check_params(args):
    assert os.path.exists(args.data_dir), ""

    args.references = os.path.join(args.data_dir, "references.txt")
    args.hypothesis = os.path.join(args.data_dir, "hypothesis.txt")

    assert os.path.isfile(args.references)
    assert os.path.isfile(args.hypothesis)


    if not hasattr(args, "src_lang_code") and not hasattr(args, 'tgt_lang_code'):
        args.src_lang_code = args.lang_code
        args.tgt_lang_code = args.lang_code

    args.evaluate_metrics = [m for m in args.evaluate_metrics.split(",") if len(m) > 0]
    args.metrics_patience = list(map(lambda x: float(x), [m for m in args.metrics_patience.split(",") if len(m) > 0]))
    assert len(args.evaluate_metrics) == len(args.metrics_patience)

def main():
    args = parse_args()
    
    check_params(args)
    global logger

    logger = initialize_exp(args)

    tokenizer = load_tokenizer(args)
    bleu, correct_paras = compute_metrics(args)
    num_setences = len(correct_paras)
    correct_paras = paras_filter_by_belu(correct_paras, bleu, args.metrics_patience[0])
    logger.info(f"file by bleu 剩下{len(correct_paras)}条句子 去掉了{num_setences-len(correct_paras)}条")
    tokenized_datasets = get_tokenized_datasets(tokenizer=tokenizer,
                                                trans_para=correct_paras,
                                                src_lang=args.lang_code,
                                                tgt_lang=args.lang_code,
                                                max_input_length=args.max_sentence_length,
                                                max_target_length=args.max_sentence_length)
    tokenized_datasets = split_datasets(tokenized_datasets, 2000)
    data_collator = get_data_collator(args, tokenizer)

    training_args = get_training_args(args)
    

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),]

    model = get_model(args) 

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(args, tokenizer),
        callbacks=callbacks,
    )

    logger.critical('training')
    if args.resume_from_checkpoint == "":
        trainer_output = trainer.train()
    else:
        logger.critical(f"接着上次的训练{args.resume_from_checkpoint}")
        trainer_output = trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    logger.info(f"trainer_output => {trainer_output}")

    logger.info(f"开始做纠正，将{args.hypothesis}里的句子纠正，并保存在{args.saved_dir+'./correct.txt'}里")
    # 这里要生成纠正文本，src就是hypo tgt是ref
    paras = get_translate_paras_from_file(args.hypothesis,args.references)
    tokenized_datasets = get_tokenized_datasets(tokenizer=tokenizer,
                                                trans_para=paras,
                                                src_lang=args.lang_code,
                                                tgt_lang=args.lang_code,
                                                max_input_length=args.max_sentence_length,
                                                max_target_length=args.max_sentence_length)
    dataloader = DataLoader(tokenized_datasets['train'], batch_size=args.batch_size, 
                            collate_fn=data_collator)
    pred, labels = [], []
    for step, x in enumerate(dataloader):
        for k in x.keys():
            x[k] = x[k].cuda()
        pred_outouts = model.generate(x['input_ids'], attention_mask = x['attention_mask'],
                                      max_length=args.max_generate_length, num_beams=args.num_beams)
        pred += tokenizer.batch_decode(pred_outouts, skip_special_tokens=True)

        if step % 10 == 0:
            logger.info(f'第{step}步翻译完成')
    pred = [x+"\n" for x in pred]

    with open(os.path.join(args.saved_dir, 'correct.txt'), 'w') as f:
        f.writelines(pred)
    logger.info("generate correct over")

main()

# fire.Fire(main)
# main(logger)

