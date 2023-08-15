# -*- coding: utf-8 -*-
import logging
import argparse
import math
import os
import sys
import random
import numpy

from sklearn import metrics
from time import strftime, localtime

from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_util import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset
from models import LSTM, ASGCN, INFORMER
from models.bert_spc import BERT_SPC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
# torch.backends.cudnn.enabled = False

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)

            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(max_seq_len=opt.max_seq_len)
            embedding_matrix = build_embedding_matrix()
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        # self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        # else:
        #     self.valset = self.testset

        if opt.device.type == 'cuda':
            print('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        # def _train(self, criterion, optimizer, train_data_loader, test_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            increase_flag = False
            # switch model to training mode
            # self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                self.model.train()
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                # inputs = [batch[col].unsqueeze(0).to(self.opt.device) for col in self.opt.inputs_cols]
                # outputs = self.model(*inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    # logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

                    val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
                    # val_acc, val_f1 = self._evaluate_acc_f1(test_data_loader)
                    print(
                        'train_loss: {:.4f}, train_acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}' \
                            .format(train_loss, train_acc, val_acc, val_f1))
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        max_val_epoch = i_epoch
                        if not os.path.exists('../user_data/state_dict'):
                            os.mkdir('../user_data/state_dict')
                        path = '../user_data/state_dict/{0}_{1}_val_acc_{2}'.format(self.opt.model_name,
                                                                                    self.opt.dataset,
                                                                                    round(val_acc, 4))
                        if max_val_acc > 0.984:
                            torch.save(self.model.state_dict(), path)
                            print('>> max_val_acc: {} has saved !'.format(max_val_acc))
                        else:
                            print('>> max_val_acc: {}'.format(max_val_acc))
                    if val_f1 > max_val_f1:
                        max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        # return path, max_val_acc, max_val_f1
        return max_val_acc, max_val_f1

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1],
                              average='macro')
        return acc, f1

    def run(self, repeats=3):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        # _params = filter(lambda p: p.requires_grad, self.model.parameters())
        # optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        if not os.path.exists('../user_data/log/'):
            os.mkdir('../user_data/log/')

        f_out = open(
            '../user_data/log/' + self.opt.model_name + '_' + self.opt.dataset + '_' + strftime("%y%m%d-%H%M",
                                                                                                localtime()) + '_val.txt',
            'w', encoding='utf-8')
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        # test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        max_test_acc_all = 0
        max_test_f1_all = 0
        for i in range(repeats):
            print('#' * 100)
            print('repeats: {}'.format(i))
            f_out.write('repeat: ' + str(i + 1))
            self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)
            # 冻结参数
            for name, param in self.model.named_parameters():
                if "bert" in name:
                    param.requires_grad = False

            # best_model_path, max_test_acc, max_test_f1 = self._train(criterion, optimizer, train_data_loader,
            #                                                          val_data_loader)
            max_test_acc, max_test_f1 = self._train(criterion, optimizer, train_data_loader,
                                                    val_data_loader)
            # best_model_path = self._train(criterion, optimizer, train_data_loader, test_data_loader)
            # self.model.load_state_dict(torch.load(best_model_path))
            # test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
            # print('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
            f_out.write('max_test_acc: {0}, max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            if max_test_acc > max_test_acc_all:
                max_test_acc_all = max_test_acc
            if max_test_f1 > max_test_f1_all:
                max_test_f1_all = max_test_f1
            print('#' * 100)
        print("max_test_acc_all:", max_test_acc_all)
        print("max_test_f1_all:", max_test_f1_all)

        f_out.close()


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='informer', type=str)
    parser.add_argument('--dataset', default='cls', type=str, help='cls')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=1e-4, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=100, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--decrease_dim', default=1, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-chinese', type=str,
                        help='distilbert-base-uncased-finetuned-sst-2-english, bert-base-uncased')

    parser.add_argument('--max_seq_len', default=1015, type=int)
    parser.add_argument('--polarities_dim', default=2, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--num_encoder_layers', default=2, type=int)
    parser.add_argument('--num_decoder_layers', default=2, type=int)
    parser.add_argument('--patience', default=5, type=int)  # early stop flag
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=776, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.2, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'lstm': LSTM,
        'informer': INFORMER,
        'bert_spc': BERT_SPC,
        'asgcn': ASGCN,

        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }
    dataset_files = {
        'cls': {
            'train': '../xfdata/ChatGPT生成文本检测器公开数据-更新/train.csv',
            'test': '../xfdata/ChatGPT生成文本检测器公开数据-更新/test.csv'
        },
    }
    input_colses = {
        'lstm': ['text_indices'],
        'informer': ['text_indices'],
        'bert_spc': ['text_bert_indices'],
        'asgcn': ['text_indices','cos_sim'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    # log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    # logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
