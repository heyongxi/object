# -*- coding: utf-8 -*-


from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=2, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.decrease_dim)
        self.dense = nn.Linear(opt.max_seq_len, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != -1, dim=-1)
        text_out, (_, _) = self.lstm(x, x_len)
        out = self.fc(text_out).squeeze(-1)

        out = self.dense(out)

        return out


# import torch
# import torch.nn as nn
#
#
# class LSTM(nn.Module):
#     def __init__(self, embedding_matrix, opt):
#         super(LSTM, self).__init__()
#
#         self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
#         self.lstm = nn.LSTM(opt.embed_dim, opt.hidden_dim, num_layers=2, bidirectional=True,
#                             dropout=opt.dropout, batch_first=True)
#         self.fc = nn.Linear(opt.hidden_dim * 2 , opt.polarities_dim)
#         self.dropout = nn.Dropout(opt.dropout)
#
#     def forward(self, text, text_lengths):
#         embedded = self.dropout(self.embedding(text))
#
#         packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
#         packed_output, (hidden, cell) = self.lstm(packed_embedded)
#
#         # 使用最后一个时间步的隐藏状态作为输出
#         if self.lstm.bidirectional:
#             hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
#         else:
#             hidden = hidden[-1, :, :]
#
#         hidden = self.dropout(hidden)
#         prediction = self.fc(hidden)
#
#         return prediction
#
#
# # # 示例用法
# # vocab_size = 10000  # 词汇表大小
# # embedding_dim = 100  # 词嵌入维度
# # hidden_size = 256  # LSTM隐藏层维度
# # output_dim = 3  # 情感极性类别数
# # num_layers = 2  # LSTM层数
# # bidirectional = True  # 是否使用双向LSTM
# # dropout = 0.5  # Dropout概率
