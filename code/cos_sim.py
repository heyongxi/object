import numpy as np
import warnings
import torch.nn as nn
import torch
import pickle
import pandas as pd
from tqdm import tqdm

from data_util import build_embedding_matrix, build_tokenizer
warnings.filterwarnings("ignore")

def cos_sim(fname):
#加载glove词嵌入，对句子进行token，然后计算单词间的余弦相似度，构建Cosine similarity图
    tokenizer = build_tokenizer(max_seq_len=1015)
    embedding_matrix = build_embedding_matrix()

    embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
    all_matrix = []

    text = pd.read_csv(fname)

    for index, row in tqdm(text.iterrows()):
        # name = row['name']
        # polarity = int(row['label'])
        sentence_tokens = row['content']

        text_indices = tokenizer.text_to_sequence(sentence_tokens)
        text_embed = embed(torch.tensor(text_indices))


        cossim_matrix = np.zeros((len(text_embed), len(text_embed))).astype('float32')
        for num1 in range(len(text_embed)):
            for num2 in range(len(text_embed)):
                cossim_matrix[num1][num2] = torch.cosine_similarity(text_embed[num1], text_embed[num2], dim=0)
        cossim_matrix[cossim_matrix<0.4] = 0
        all_matrix.append(cossim_matrix)
    print(len(all_matrix))

    if 'train' in fname:
        f = open('../user_data/raw_cos_train.graph', 'wb')
        pickle.dump(all_matrix, f)
    else:
        f = open('../user_data/raw_cos_test.graph', 'wb')
        pickle.dump(all_matrix, f)

if __name__ == '__main__':
    cos_sim('../xfdata/ChatGPT生成文本检测器公开数据-更新/train.csv')
    cos_sim('../xfdata/ChatGPT生成文本检测器公开数据-更新/test.csv')