"""
在 BertForSequenceClassification 所在的文件中自己定义了 MyBertForSequenceClassification 用于对它进行自定义!!!
"""
import torch 
import argparse
import numpy as np
import copy
import sklearn
import scipy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import required
from torch.optim import optimizer
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import BertForSequenceClassification
from transformers import MyBertForSequenceClassification
from transformers.optimization import AdamW
from torch.utils import data
from pytorch_pretrained_bert import BertModel
# data.DataLoader
print(required)

# Let's encode some text in a sequence of hidden-states using each model:
    # Load pretrained model/tokenizer
pretrained_weights = 'bert-base-chinese'

torch.manual_seed(40)
# 这里的 from_pretrained 是一个类方法，所以 BertTokenizer 类和 BertModel 类可以调用
# 返回值是 对应类的实例，实例 tokenizer 属性中有 字典 vocab ，还有分词方法，具体可以查看源码
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
# model = BertModel.from_pretrained(pretrained_weights)
model = BertForSequenceClassification.from_pretrained(pretrained_weights, num_labels=4)
# model = model.eval()
# print(model)
# print(list(tokenizer.vocab.items())[2000:2005])
# Encode text
# Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
input_ids = torch.tensor([tokenizer.encode("京  东,商 品真 是不错", add_special_tokens=True)]) 
labels = torch.LongTensor([1]).unsqueeze(0)
# last_hidden_states = model(input_ids, labels=labels)  # Models outputs are now tuples

# last_hidden_states = model(input_ids)
# print(last_hidden_states[0])

