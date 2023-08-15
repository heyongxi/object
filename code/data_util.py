import os

import numpy as np
import pickle
import re
from torch.utils.data import Dataset

import pandas as pd
from transformers import BertTokenizer

# 解析content中的数字
def parse_content(content):
    tokens = re.findall(r'\d+', content)
    return [int(token) for token in tokens]


def build_embedding_matrix():
    with open("../user_data/embedding_matrix.pkl", "rb") as f:
        embedding_matrix = pickle.load(f)
    return embedding_matrix

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


def build_tokenizer(max_seq_len):

    tokenizer = Tokenizer(max_seq_len)

    return tokenizer

class Tokenizer(object):
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len


    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = parse_content(text)
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        # sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text)[:self.max_seq_len - 2] + ['[SEP]']
        sequence = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)



class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        all_data = []
        text = pd.read_csv(fname)
        fin = open('../user_data/raw_cos_train.graph', 'rb')
        cos_sim_graph = pickle.load(fin)
        fin.close()

        for index, row in text.iterrows():
            name = row['name']
            polarity = int(row['label'])
            sentence_tokens = row['content']

            sentence = '[CLS] ' + sentence_tokens + ' [SEP]'
            cos_sim = cos_sim_graph[name-1]
            text_indices = tokenizer.text_to_sequence(sentence_tokens)
            text_bert_indices = tokenizer.text_to_sequence(sentence)

            data = {
                'text_bert_indices': text_bert_indices,
                'name': name,
                'text_indices': text_indices,
                'cos_sim': cos_sim,
                'polarity': polarity,
            }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)