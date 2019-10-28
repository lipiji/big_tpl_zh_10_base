import random
import torch
import numpy as np
import re

PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'
BOC, EOC = '<boc>', '<eoc>'
LS, RS, SP = '<s>', '</s>', ' '
CS = ['<c-1>'] + ['<c' + str(i) + '>' for i in range(32)]
SS = ['<s-1>'] + ['<s' + str(i) + '>' for i in range(512)]
PS = ['<p-1>'] + ['<p' + str(i) + '>' for i in range(512)]

BUFSIZE = 4096000

def cut_sent(para):
    para = re.sub('([，。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")

def ListsToTensor(xs, vocab=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if vocab is not None:
            y = vocab.token2idx(x) + [vocab.padding_idx]*(max_len - len(x))
        else:
            y = x + [0]*(max_len - len(x))
        ys.append(y)
    return ys

def _back_to_text_for_check(x, vocab):
    w = x.t().tolist()
    for sent in vocab.idx2token(w):
        print (' '.join(sent))
    
def batchify(data, vocab):
    ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk = [], [], [], [], [], []
    for ys_i, ys_tpl_i, ys_seg_i, ys_pos_i in data:
        ys_truth.append(ys_i)
        ys_inp.append([BOS] + ys_i[:-1])
        ys_tpl.append(ys_tpl_i)
        ys_seg.append(ys_seg_i)
        ys_pos.append(ys_pos_i)
        msk.append([1 for i in range(len(ys_i))])

    ys_truth = torch.LongTensor(ListsToTensor(ys_truth, vocab)).t_().contiguous()
    ys_inp = torch.LongTensor(ListsToTensor(ys_inp, vocab)).t_().contiguous()
    ys_tpl = torch.LongTensor(ListsToTensor(ys_tpl, vocab)).t_().contiguous()
    ys_seg = torch.LongTensor(ListsToTensor(ys_seg, vocab)).t_().contiguous()
    ys_pos = torch.LongTensor(ListsToTensor(ys_pos, vocab)).t_().contiguous()
    msk = torch.LongTensor(ListsToTensor(msk)).t_().contiguous().to(torch.uint8)
    return ys_truth, ys_inp, ys_tpl, ys_seg, ys_pos, msk

def random_mask(tokens, masked_prob=0.8):
    num_to_predict = max(1, int(round(len(tokens) * masked_prob)))
    cand = []
    for i, token in enumerate(tokens):
        cand.append(i)
    random.shuffle(cand)
    cand = set(cand[:num_to_predict])

    masked_tokens, mask = [], []
    for i, token in enumerate(tokens):
        if i in cand:
            masked_tokens.append(CS[1])
            mask.append(1)
        else:
            masked_tokens.append(token)
            mask.append(0)
    return masked_tokens, mask


def prepare_sample(tokens):
    ys = []
    ys_tpl = []
    ys_seg = []
    ys_pos = []

    sents = cut_sent(''.join(tokens))
    segi = 0
    for sent in sents:
        if not sent:
            continue
        segi += 1
        
        ws = [w for w in sent]
        masked_ws, msk = random_mask(ws, 0.8)
        
        ys += ws + [RS]
        ys_tpl += masked_ws + [RS]
        ys_seg += [SS[segi] for w in ws] + [RS]
        ys_pos += [PS[i + 1] for i in range(len(ws))] + [RS]

    return ys + [EOS], ys_tpl + [EOS], ys_seg + [EOS], ys_pos + [EOS]

def s2t(strs, vocab):
    inp, msk = [], []
    for x in strs:
        inp.append([w for w in x])
        msk.append([1 for i in range(len(x))])

    inp = torch.LongTensor(ListsToTensor(inp, vocab)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return inp, msk

class DataLoader(object):
    def __init__(self, vocab, filename, batch_size, max_len, min_len):
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_len = max_len
        self.min_len = min_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

    def __iter__(self):
        
        lines = self.stream.readlines(BUFSIZE)

        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        data = []
        for line in lines[:-1]: #the last sent may be imcomplete
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < self.min_len:
                continue
                
            data.append(prepare_sample(tokens))
        random.shuffle(data)
        
        idx = 0
        while idx < len(data):
            yield batchify(data[idx:idx+self.batch_size], self.vocab)
            idx += self.batch_size

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        idx2token = [PAD, UNK, BOS, EOS] + [BOC, EOC, LS + RS + SP] + CS + SS + PS \
                    +  (specials if specials is not None else [])
        for line in open(filename, encoding='utf8').readlines():
            try: 
                token, cnt = line.strip().split()
            except:
                continue
            if int(cnt) >= min_occur_cnt:
                idx2token.append(token)
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    @property
    def size(self):
        return len(self._idx2token)
    
    @property
    def unk_idx(self):
        return self._unk_idx
    
    @property
    def padding_idx(self):
        return self._padding_idx
    
    def random_token(self):
        return self.idx2token(1 + np.random.randint(self.size-1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)
