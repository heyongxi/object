# -*- coding: utf-8 -*-
import csv
import os
import pickle
from time import strftime, localtime

import pandas as pd
import torch
import torch.nn.functional as F
import argparse
import numpy as np

from data_util import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, pad_and_truncate
from models import LSTM, ASGCN, INFORMER
from models.bert_spc import BERT_SPC

from transformers import BertModel


class Inferer:
    """A simple inference example"""

    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)

            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            self.tokenizer = build_tokenizer(max_seq_len=opt.max_seq_len)
            embedding_matrix = build_embedding_matrix()

            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, name, text, cos_sim):

        text_indices = torch.tensor(self.tokenizer.text_to_sequence(text))
        # text_bert_indices = torch.tensor(self.tokenizer.text_to_sequence(text))

        data = {
            # 'text_bert_indices': text_bert_indices,
            'text_indices': text_indices,
            'cos_sim': cos_sim,
            # 'name':name,
        }

        t_inputs = [data[col].unsqueeze(0).to(self.opt.device) for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)
        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()

        return t_probs, name


if __name__ == '__main__':
    model_classes = {
        'lstm': LSTM,
        'informer': INFORMER,
        'bert_spc': BERT_SPC,
        'asgcn': ASGCN,
    }
    dataset_files = {
        'cls': {
            # 'train': '../xfdata/ChatGPT生成文本检测器公开数据-更新/train.csv',
            'test': '../xfdata/ChatGPT生成文本检测器公开数据-更新/test.csv',
        },
    }
    input_colses = {
        'lstm': ['text_indices'],
        'informer': ['text_indices'],
        'bert_spc': ['concat_bert_indices'],
        'asgcn': ['text_indices', 'cos_sim'],
    }


    class Option(object):
        pass


    opt = Option()
    opt.model_name = 'lstm'
    opt.model_class = model_classes[opt.model_name]
    opt.dataset = 'cls'
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    # set your trained models here
    opt.state_dict_path = '../user_data/state_dict/lstm_cls_val_acc_0.9904'
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 1015
    opt.bert_dim = 768
    opt.pretrained_bert_name = 'bert-base-uncased'
    opt.polarities_dim = 2
    opt.hops = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.local_context_focus = 'cdm'
    opt.SRD = 3
    opt.decrease_dim = 1
    opt.num_heads = 2
    opt.num_encoder_layers = 2
    opt.num_decoder_layers = 2

    inf = Inferer(opt)
    text = pd.read_csv(opt.dataset_file['test'])
    fin = open('../user_data/raw_cos_train.graph', 'rb')
    cos_sim_graph = pickle.load(fin)
    fin.close()
    if not os.path.exists('../prediction_result/'):
        os.mkdir('../prediction_result/')
    with open(
            '../prediction_result/' + opt.model_name + '_' + opt.dataset + '_' + strftime("%y%m%d-%H%M",
                                                                                          localtime()) + '_result.csv',
            'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "label"])
        for index, row in text.iterrows():
            name = row['name']
            sentence_tokens = row['content']
            cos_sim = torch.tensor(cos_sim_graph[name - 14001])
            t_probs, name = inf.evaluate(name, sentence_tokens, cos_sim)
            writer.writerow([name, t_probs.argmax(axis=-1)[0]])
            # print(t_probs.argmax(axis=-1))
