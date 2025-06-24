import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端
import pandas as pd
import jieba
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from gensim.models import Word2Vec

# 设置matplotlib的字体，这里使用SimHei字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题

# 1. 数据加载与预处理
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['text'] = df['evaluation'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # 清洗文本
    df['tokens'] = df['text'].apply(jieba.lcut)  # 分词
    df['label'] = df['label'].map({'正面': 1, '负面': 0})  # 标签映射
    return df

# 2. 词向量训练与序列处理
def prepare_sequences(df, max_len=100):
    texts = df['tokens'].tolist()
    # Word2Vec训练
    w2v_model = Word2Vec(texts, vector_size=100, window=5, min_count=1, workers=4)
    vocab_size = len(w2v_model.wv.index_to_key)
    word2idx = {word: i for i, word in enumerate(w2v_model.wv.index_to_key)}

    # 转换为索引序列并padding
    texts_idx = [[word2idx[word] for word in text] for text in texts]
    X = pad_sequences(texts_idx, maxlen=max_len, padding='post')
    y = df['label'].values

    # 嵌入矩阵
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in word2idx.items():
        embedding_matrix[i] = w2v_model.wv[word]

    return X, y, embedding_matrix, vocab_size, max_len

# 3. 构建LSTM模型
def build_lstm_model(vocab_size, max_len, embedding_matrix):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# 4. 训练与评估
def train_evaluate(model, X, y, test_size=0.2, epochs=10, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # 评估
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试集准确率: {acc:.4f}")

    # 分类报告
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=['负面', '正面']))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('情感分析混淆矩阵')
    plt.show()

    return history

# 5. 可视化训练历史
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('准确率变化')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    file_path = 'data_single.csv'
    df = load_data(file_path)
    X, y, embedding_matrix, vocab_size, max_len = prepare_sequences(df, max_len=100)
    model = build_lstm_model(vocab_size, max_len, embedding_matrix)
    history = train_evaluate(model, X, y, epochs=10, batch_size=32)
    plot_history(history)