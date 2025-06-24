import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ====================== 环境配置（解决中文显示和后端问题） ======================
import matplotlib

# 切换后端（根据环境选择，优先尝试'TkAgg'，若报错则换'QtAgg'或'Agg'）
matplotlib.use('TkAgg')  # Windows建议'TkAgg'，Linux/Mac可试'QtAgg'
plt.rcParams['font.sans-serif'] = ['SimSun']  # 中文支持（Windows）
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题


# ====================== 1. 数据加载与预处理 ======================
def load_and_preprocess_data(file_path, text_column=None, label_column=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")

    data = pd.read_csv(file_path)

    # 自动检测文本列
    if text_column is None:
        possible_text = ['evaluation', 'text', '评论', '评论文本', 'content']
        for col in possible_text:
            if col in data.columns:
                text_column = col
                break
    if text_column is None:
        raise ValueError("未检测到文本列！请手动指定。")

    # 自动检测标签列
    if label_column is None:
        possible_labels = ['Label', 'label', '情感标签', '情绪', '分类']
        for col in possible_labels:
            if col in data.columns:
                label_column = col
                break
    if label_column is None:
        raise ValueError("未检测到标签列！请手动指定。")

    print(f"文本列: {text_column} | 标签列: {label_column}")
    print("\n标签分布:")
    print(data[label_column].value_counts())

    # 自动映射标签（支持"正面/负面"等常见表述）
    label_map = {}
    unique_labels = data[label_column].unique()
    if '正面' in unique_labels and '负面' in unique_labels:
        label_map = {'正面': 1, '负面': 0}
    elif '积极' in unique_labels and '消极' in unique_labels:
        label_map = {'积极': 1, '消极': 0}
    else:
        raise ValueError(f"无法自动映射标签！发现标签: {unique_labels}")

    data['Label'] = data[label_column].map(label_map)
    data = data.dropna(subset=['Label'])  # 过滤无效标签

    print(f"\n有效样本数: {len(data)} | 正面比例: {data['Label'].mean():.2%}")
    return data, text_column


# ====================== 2. 加载BERT模型和Tokenizer ======================
def load_bert_model(model_path='bert-base-chinese'):
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        bert_model = TFAutoModel.from_pretrained(model_path)
        print("BERT模型加载成功！")
        return tokenizer, bert_model
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise


# ====================== 3. 文本编码 ======================
def encode_texts(texts, tokenizer, max_seq_len=128):
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_seq_len,
        return_tensors='tf'
    )


# ====================== 4. 构建带注意力机制的模型 ======================
def build_model(bert_model, max_seq_len=128, num_heads=2):
    input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='attention_mask')

    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]

    key_dim = bert_output.shape[-1] // num_heads
    attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name='multi_head_attention'
    )
    attention_output = attention_layer(bert_output, bert_output, bert_output)

    x = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    for layer in bert_model.layers:
        layer.trainable = False  # 冻结BERT层

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ====================== 5. 训练过程可视化 ======================
def visualize_training(history):
    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练集')
    plt.plot(history.history['val_accuracy'], label='验证集')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('准确率')
    plt.legend()
    plt.title('模型准确率')

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练集')
    plt.plot(history.history['val_loss'], label='验证集')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('损失')
    plt.legend()
    plt.title('模型损失')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("训练指标已保存为: training_metrics.png")
    plt.show()
    plt.close()  # 释放资源


# ====================== 6. 模型评估 ======================
def evaluate_model(model, test_dataset, test_labels):
    test_pred_proba = model.predict(test_dataset).flatten()
    test_pred_labels = (test_pred_proba > 0.5).astype(int)

    print("\n分类报告:")
    print(classification_report(
        test_labels, test_pred_labels, target_names=['负面', '正面']
    ))

    cm = confusion_matrix(test_labels, test_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存为: confusion_matrix.png")
    plt.show()
    plt.close()

    return test_pred_proba


# ====================== 7. 注意力可视化 ======================
def visualize_attention(model, tokenizer, sample_text, max_seq_len=128):
    print(f"\n=== 注意力可视化 ===\n文本: {sample_text[:80]}...")

    sample_encoding = tokenizer(
        sample_text, truncation=True, padding='max_length',
        max_length=max_seq_len, return_tensors='tf'
    )

    # 提取注意力分数
    attention_model = tf.keras.Model(
        inputs=[model.input[0], model.input[1]],
        outputs=model.get_layer('multi_head_attention').attention_scores
    )
    attention_scores = attention_model.predict({
        'input_ids': sample_encoding['input_ids'],
        'attention_mask': sample_encoding['attention_mask']
    })

    # 处理有效Token
    valid_mask = sample_encoding['attention_mask'][0].numpy()
    valid_length = np.sum(valid_mask)
    valid_tokens = tokenizer.convert_ids_to_tokens(
        sample_encoding['input_ids'][0].numpy()[:valid_length]
    )

    # 可视化第一个注意力头
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_scores[0, 0, :valid_length, :valid_length],
               cmap='hot', interpolation='nearest')
    plt.xticks(np.arange(valid_length), valid_tokens, rotation=90)
    plt.yticks(np.arange(valid_length), valid_tokens)
    plt.title("注意力热力图（第1头）")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('attention_heatmap.png')
    print("注意力热力图已保存为: attention_heatmap.png")
    plt.show()
    plt.close()

    # 预测结果
    pred_proba = model.predict(sample_encoding)[0][0]
    pred_label = '正面' if pred_proba > 0.5 else '负面'
    print(f"预测: {pred_label}（概率: {pred_proba:.4f}）")


# ====================== 主程序 ======================
def main():
    # 配置（根据实际路径调整）
    DATA_FILE = 'data_single.csv'  # 数据集路径
    BERT_PATH = 'bert-base-chinese'  # BERT模型路径
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3

    # 1. 数据加载
    try:
        data, text_col = load_and_preprocess_data(DATA_FILE)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 2. 划分数据集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data[text_col], data['Label'], test_size=0.2, random_state=42
    )

    # 3. 加载BERT
    try:
        tokenizer, bert_model = load_bert_model(BERT_PATH)
    except Exception as e:
        print(f"BERT加载失败: {e}")
        return

    # 4. 编码文本
    train_enc = encode_texts(train_texts, tokenizer, MAX_SEQ_LEN)
    test_enc = encode_texts(test_texts, tokenizer, MAX_SEQ_LEN)

    # 5. 构建Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((
        {k: v for k, v in train_enc.items()}, train_labels
    )).shuffle(1000).batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices((
        {k: v for k, v in test_enc.items()}, test_labels
    )).batch(BATCH_SIZE)

    # 6. 构建并训练模型
    model = build_model(bert_model, MAX_SEQ_LEN)
    print("\n模型结构:")
    model.summary()

    print("\n开始训练...")
    history = model.fit(
        train_ds, epochs=EPOCHS, validation_data=test_ds
    )

    # 7. 可视化训练过程
    visualize_training(history)

    # 8. 评估模型
    evaluate_model(model, test_ds, test_labels)

    # 9. 注意力可视化（随机选测试样本）
    sample_idx = np.random.randint(0, len(test_texts))
    visualize_attention(model, tokenizer, test_texts.iloc[sample_idx], MAX_SEQ_LEN)

    # 10. 保存模型
    model.save('sentiment_model')
    print("\n模型已保存为: sentiment_model")


if __name__ == "__main__":
    main()