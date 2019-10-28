# coding=utf-8
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from biglm import BIGLM
from data import Vocab, DataLoader
from adam import AdamWeightDecayOptimizer
from optim import Optim

import argparse, os
import random


torch.manual_seed(1234)
vocab = Vocab("./data/vocab.txt", min_occur_cnt=200, specials=[])
print (vocab.size, vocab.padding_idx)

train_data = DataLoader(vocab, "./data/train.txt06", 20, 512)
batch_acm = 0
while True:
    for truth, inp, msk in train_data:
        batch_acm += 1
        #print(batch_acm)
        #if batch_acm <= 123599:
        #    continue
        #print(torch.sum(truth, 0), torch.sum(inp, 0), torch.sum(msk, 0))    
        summ = torch.sum(msk, 0)
        min_len = torch.min(summ)
        if min_len < 20:
            print(min_len)
        #n = torch.nonzero(summ).size(0)
        #if n < 20:
        #    print(n, batch_acm)
    if train_data.epoch_id >= 1:
        break
