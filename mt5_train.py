
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          MT5ForConditionalGeneration, AutoTokenizer,
                          DataCollatorForSeq2Seq)

import os
import torch
import logging


## 准备数据集