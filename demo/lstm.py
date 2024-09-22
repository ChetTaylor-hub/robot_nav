# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

# 超参数设置
data_path = './data/modified_robot_commands_100k.txt'  # 数据集
glove_path = './data/glove.6B.50d.txt'     # Glove词向量文件路径
save_path = './saved_dict/lstm.ckpt'       # 模型训练结果

# 读取Glove词向量函数
def load_glove_embeddings(glove_path):
    vocab = {}
    embeddings = []

    with open(glove_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            vocab[word] = idx
            embeddings.append(vector)
    
    embeddings = np.stack(embeddings)
    return vocab, embeddings

# 加载Glove词向量和词表
vocab, embeddings = load_glove_embeddings(glove_path)
embedding_pretrained = torch.tensor(embeddings)

embed = embedding_pretrained.size(1)        # 词向量维度
dropout = 0.5                               # 随机丢弃
num_classes = 4                             # 类别数
num_epochs = 10                             # epoch数
batch_size = 128                            # mini-batch大小
pad_size = 50                               # 每句话处理成的长度(短填长切)
learning_rate = 1e-3                        # 学习率
hidden_size = 128                           # lstm隐藏层
num_layers = 2                              # lstm层数
MAX_VOCAB_SIZE = 400000                      # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'                 # 未知字，padding符号


def get_data():
    tokenizer = lambda x: [y for y in x]  # 字级别
    print('vocab', vocab)
    print(f"Vocab size: {len(vocab)}")

    train, dev, test = load_dataset(data_path, pad_size, tokenizer, vocab)
    return vocab, train, dev, test

# 修改后的标签映射
label_map = {
    'horse': 0,
    'banana': 1,
    'bus': 2,
    'bottle': 3
}

def load_dataset(path, pad_size, tokenizer, vocab):
    contents = []
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            label, content = lin.split('	####	')
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD, 0)] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK, 0)))
            label = label_map.get(label, -1)
            if label != -1:
                contents.append((words_line, int(label)))
                label_counts[label] += 1  # 统计每个标签的数量

    print("Label distribution:", label_counts)

    train, X_t = train_test_split(contents, test_size=0.4, random_state=42)
    dev, test = train_test_split(X_t, test_size=0.5, random_state=42)
    return train, dev, test


class TextDataset(Dataset):
    def __init__(self, data):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
        self.y = torch.LongTensor([x[1] for x in data]).to(self.device)
    def __getitem__(self, index):
        self.text = self.x[index]
        self.label = self.y[index]
        return self.text, self.label
    def __len__(self):
        return len(self.x)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 定义LSTM模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 使用预训练的词向量模型，freeze=False 表示允许参数在训练中更新
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        self.lstm = nn.LSTM(embed, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)

def plot_acc(train_acc):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_acc)))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(loc='best')
    plt.savefig('results/acc.png', dpi=400)

def plot_loss(train_loss):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('results/loss.png', dpi=400)

def train(model, dataloaders):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    dev_best_loss = float('inf')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Start Training...\n")
    plot_train_acc = []
    plot_train_loss = []

    for i in range(num_epochs):
        step = 0
        train_lossi = 0
        train_acci = 0
        for inputs, labels in dataloaders['train']:
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            step += 1
            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_lossi += loss.item()
            train_acci += metrics.accuracy_score(true, predic)

        dev_acc, dev_loss = dev_eval(model, dataloaders['dev'], loss_function, Result_test=False)
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), save_path)
        train_acc = train_acci / step
        train_loss = train_lossi / step
        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)
        print(f"epoch = {i+1} :  train_loss = {train_loss:.3f}, train_acc = {train_acc:.2%}, dev_loss = {dev_loss:.3f}, dev_acc = {dev_acc:.2%}")
    
    plot_acc(plot_train_acc)
    plot_loss(plot_train_loss)

    model.load_state_dict(torch.load(save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = dev_eval(model, dataloaders['test'], loss_function, Result_test=True)
    print('================' * 8)
    print(f'test_loss: {test_loss:.3f}      test_acc: {test_acc:.2%}')

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

def result_test(real, pred):
    # 确保使用完整标签
    labels11 = ['horse', 'banana', 'bus', 'bottle']
    # 生成完整标签集的混淆矩阵
    cv_conf = confusion_matrix(real, pred, labels=list(range(len(labels11))))
    # 计算准确率、精度、召回率和 F1 分数
    acc = accuracy_score(real, pred)
    precision = precision_score(real, pred, average='micro')
    recall = recall_score(real, pred, average='micro')
    f1 = f1_score(real, pred, average='micro')
    print(f'test:  acc: {acc:.4f}   precision: {precision:.4f}   recall: {recall:.4f}   f1: {f1:.4f}')
    # 使用完整的标签列表
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
    # 绘制混淆矩阵
    disp.plot(cmap="Blues", values_format='')
    # 保存图片
    plt.savefig("results/reConfusionMatrix.tif", dpi=100)


def dev_eval(model, data, loss_function, Result_test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data:
            outputs = model(texts)
            loss = loss_function(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if Result_test:
        result_test(labels_all, predict_all)
    return acc, loss_total / len(data)

if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = get_data()
    dataloaders = {
        'train': DataLoader(TextDataset(train_data), batch_size, shuffle=True),
        'dev': DataLoader(TextDataset(dev_data), batch_size, shuffle=True),
        'test': DataLoader(TextDataset(test_data), batch_size, shuffle=True)
    }
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Model().to(device)
    init_network(model)
    train(model, dataloaders)
