import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import re
import pickle

# 读取CSV文件
train_df = pd.read_csv("../xfdata/ChatGPT生成文本检测器公开数据-更新/train.csv")
test_df = pd.read_csv("../xfdata/ChatGPT生成文本检测器公开数据-更新/test.csv")

# 合并训练集和测试集以便共同构建词向量
combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# 定义函数来解析content列中的数字
def parse_content(content):
    tokens = re.findall(r'\d+', content)
    return [int(token) for token in tokens]

# 解析content列并创建一个新的列存储解析后的tokens
combined_df['parsed_content'] = combined_df['content'].apply(parse_content)

# 使用TF-IDF向量化tokens
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_df['parsed_content'])

# 将TF-IDF矩阵转换为稠密数组
tfidf_array = tfidf_matrix.toarray()

# 初始化Word2Vec模型并构建词向量
w2v_model = Word2Vec(vector_size=300, min_count=1)
w2v_model.build_vocab_from_freq(tfidf_vectorizer.vocabulary_)
w2v_model.train(combined_df['parsed_content'], total_examples=len(combined_df['parsed_content']), epochs=10)

# 创建embedding_matrix并填充词向量
embedding_matrix = np.zeros((len(tfidf_vectorizer.vocabulary_), 300))
for token, token_id in tfidf_vectorizer.vocabulary_.items():
    embedding_matrix[token_id] = w2v_model.wv[token]

# 保存embedding_matrix为pkl文件
with open("../user_data/embedding_matrix.pkl", "wb") as f:
    pickle.dump(embedding_matrix, f)

print("Embedding matrix saved to embedding_matrix.pkl")
