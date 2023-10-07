from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import os

from transformers import (HfArgumentParser,
                          Seq2SeqTrainingArguments,
                          AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          Seq2SeqTrainer,
                          DataCollatorForSeq2Seq)

from torch.utils.data import Dataset

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="mt5-small")


@dataclass
class DataArguments:
    src_file: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    tgt_file: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    model_max_length: int = field(default=256, metadata={"help": "Maximum sequence length."})

    src_lang: str = field(default="en", metadata={"help": "Source language"})
    tgt_lang: str = field(default="zh", metadata={"help": "Target language"})


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        args,
        tokenizer,
    ):
        super(SupervisedDataset, self).__init__()

        self.src_lang = args.src_lang
        self.tgt_lang = args.tgt_lang

        self.src_file = args.src_file.split(",")[0]
        self.tgt_file = args.tgt_file.split(",")[0]
        
        def _read(file):
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return [s.strip("\n") for s in lines]
        self.data = list(zip(_read(self.src_file), _read(self.tgt_file)))
        
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length

        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):

        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang

        model_inputs = self.tokenizer(example[0], max_length = self.model_max_length,truncation=True)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(example[1], max_length = self.model_max_length,truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=data_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    
    dataset = SupervisedDataset(args=data_args, tokenizer=tokenizer)
    #Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, model=model, max_length=data_args.model_max_length)
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
